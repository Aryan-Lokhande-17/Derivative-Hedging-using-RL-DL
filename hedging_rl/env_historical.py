from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass(slots=True)
class HistoricalState:
    step: int
    spot: float
    hedge_position: float
    cash_account: float


class HistoricalDerivativeHedgingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Hedging environment that replays historical spot prices (e.g., LSE)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.Series,
        strike: float,
        rate: float,
        maturity_steps: int,
        transaction_cost: float,
        max_position: float = 2.0,
        risk_aversion: float = 1.0,
    ) -> None:
        super().__init__()
        self.prices = prices.astype(float).reset_index(drop=True)
        self.strike = float(strike)
        self.rate = float(rate)
        self.maturity_steps = int(maturity_steps)
        self.transaction_cost = float(transaction_cost)
        self.max_position = max_position
        self.risk_aversion = risk_aversion

        if len(self.prices) <= self.maturity_steps:
            raise ValueError("Price series must be longer than maturity_steps")

        self.action_space = spaces.Box(
            low=np.array([-max_position], dtype=np.float32),
            high=np.array([max_position], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, -10.0, 0.0, -max_position], dtype=np.float32),
            high=np.array([1.0, 10.0, 10.0, max_position], dtype=np.float32),
            dtype=np.float32,
        )

        self.start_idx = 0
        self.state: HistoricalState | None = None

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment state not initialized. Call reset().")
        tau_ratio = max((self.maturity_steps - self.state.step) / self.maturity_steps, 0.0)
        log_moneyness = np.log((self.state.spot + 1e-12) / self.strike)
        normalized_spot = self.state.spot / self.prices.iloc[self.start_idx]
        return np.array([tau_ratio, log_moneyness, normalized_spot, self.state.hedge_position], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        max_start = len(self.prices) - self.maturity_steps - 1
        if max_start <= 0:
            self.start_idx = 0
        else:
            self.start_idx = int(self.np_random.integers(0, max_start))

        initial_spot = float(self.prices.iloc[self.start_idx])
        self.state = HistoricalState(step=0, spot=initial_spot, hedge_position=0.0, cash_account=0.0)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if self.state is None:
            raise RuntimeError("Environment state not initialized. Call reset().")

        target_position = float(np.clip(action[0], -self.max_position, self.max_position))
        trade = target_position - self.state.hedge_position
        trade_notional = trade * self.state.spot
        cost = self.transaction_cost * abs(trade_notional)

        cash = self.state.cash_account - trade_notional - cost

        next_step = self.state.step + 1
        next_spot = float(self.prices.iloc[self.start_idx + next_step])

        dt = 1.0 / 252.0
        cash *= np.exp(self.rate * dt)
        portfolio = target_position * next_spot + cash

        done = next_step >= self.maturity_steps
        if done:
            liability = max(next_spot - self.strike, 0.0)
        else:
            liability = 0.0

        error = portfolio - liability
        reward = -self.risk_aversion * (error**2) - cost

        self.state = HistoricalState(step=next_step, spot=next_spot, hedge_position=target_position, cash_account=cash)
        info = {"replication_error": error, "transaction_cost": cost, "portfolio": portfolio, "liability": liability}
        return self._get_obs(), float(reward), done, False, info
