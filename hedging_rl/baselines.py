from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MarketConfig
from .pricing import black_scholes_call_delta, black_scholes_call_price


@dataclass(slots=True)
class HedgeRunResult:
    final_error: float
    transaction_costs: float


def run_delta_hedge(path: np.ndarray, config: MarketConfig) -> HedgeRunResult:
    dt = config.dt
    cash = black_scholes_call_price(path[0], config.K, config.T, config.r, config.sigma)
    position = 0.0
    total_cost = 0.0

    for step in range(config.steps):
        tau = max(config.T - step * dt, 0.0)
        target = black_scholes_call_delta(path[step], config.K, tau, config.r, config.sigma)
        trade = target - position
        notional = trade * path[step]
        cost = config.transaction_cost * abs(notional)
        cash -= notional + cost
        total_cost += cost

        cash *= np.exp(config.r * dt)
        position = target

    terminal_portfolio = position * path[-1] + cash
    payoff = max(path[-1] - config.K, 0.0)
    return HedgeRunResult(final_error=float(terminal_portfolio - payoff), transaction_costs=float(total_cost))
