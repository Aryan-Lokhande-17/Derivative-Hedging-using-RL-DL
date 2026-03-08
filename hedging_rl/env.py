from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import MarketConfig
from .market import GBMSimulator
from .pricing import black_scholes_call_delta, black_scholes_call_price


@dataclass(slots=True)
class HedgingState:
    step: int
    spot: float
    hedge_position: float
    cash_account: float
    option_value: float


class DerivativeHedgingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-option hedging environment for RL agents.

    The agent is short one European call option and chooses stock hedge positions.
    Reward is the negative of squared replication error and transaction costs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: MarketConfig,
        max_position: float = 2.0,
        risk_aversion: float = 1.0,
        cost_penalty: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.max_position = max_position
        self.risk_aversion = risk_aversion
        self.cost_penalty = cost_penalty

        self.action_space = spaces.Box(
            low=np.array([-max_position], dtype=np.float32),
            high=np.array([max_position], dtype=np.float32),
            dtype=np.float32,
        )

        # [time_to_maturity, log_moneyness, normalized_spot, current_position, bs_delta]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -10.0, 0.0, -max_position, 0.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 10.0, max_position, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        self.simulator = GBMSimulator(config=config, rng=self.rng)
        self.state: HedgingState | None = None

    def _option_value(self, spot: float, step: int) -> float:
        tau = max(self.config.T - step * self.config.dt, 0.0)
        return black_scholes_call_price(spot, self.config.K, tau, self.config.r, self.config.sigma)

    def _option_delta(self, spot: float, step: int) -> float:
        tau = max(self.config.T - step * self.config.dt, 0.0)
        return black_scholes_call_delta(spot, self.config.K, tau, self.config.r, self.config.sigma)

    def _portfolio_value(self, spot: float, hedge_position: float, cash_account: float) -> float:
        return hedge_position * spot + cash_account

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment state not initialized. Call reset().")

        tau_ratio = max((self.config.steps - self.state.step) / self.config.steps, 0.0)
        log_moneyness = np.log((self.state.spot + 1e-12) / self.config.K)
        normalized_spot = self.state.spot / self.config.S0
        delta = self._option_delta(self.state.spot, self.state.step)

        obs = np.array(
            [tau_ratio, log_moneyness, normalized_spot, self.state.hedge_position, delta],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.simulator = GBMSimulator(config=self.config, rng=self.rng)

        spot = self.config.S0
        option_value = self._option_value(spot, step=0)

        # Short one option, so initial cash comes from selling it.
        self.state = HedgingState(step=0, spot=spot, hedge_position=0.0, cash_account=option_value, option_value=option_value)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment state not initialized. Call reset().")

        target_position = float(np.clip(action[0], -self.max_position, self.max_position))
        previous_position = self.state.hedge_position
        traded_shares = target_position - previous_position

        trade_notional = traded_shares * self.state.spot
        transaction_cost = self.config.transaction_cost * abs(trade_notional)

        # Rebalance portfolio at current spot.
        new_cash = self.state.cash_account - trade_notional - transaction_cost

        next_step = self.state.step + 1
        next_spot = self.simulator.sample_next_price(self.state.spot)

        new_cash *= np.exp(self.config.r * self.config.dt)
        portfolio_value = self._portfolio_value(next_spot, target_position, new_cash)

        done = next_step >= self.config.steps
        if done:
            option_liability = max(next_spot - self.config.K, 0.0)
        else:
            option_liability = self._option_value(next_spot, next_step)

        replication_error = portfolio_value - option_liability
        reward = -self.risk_aversion * (replication_error**2) - self.cost_penalty * transaction_cost

        self.state = HedgingState(
            step=next_step,
            spot=next_spot,
            hedge_position=target_position,
            cash_account=new_cash,
            option_value=option_liability,
        )

        info = {
            "transaction_cost": transaction_cost,
            "replication_error": replication_error,
            "portfolio_value": portfolio_value,
            "option_liability": option_liability,
        }

        return self._get_obs(), float(reward), done, False, info
