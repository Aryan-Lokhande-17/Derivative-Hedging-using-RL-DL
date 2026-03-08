from __future__ import annotations

import numpy as np

from .config import MarketConfig


class GBMSimulator:
    """Geometric Brownian Motion path simulator."""

    def __init__(self, config: MarketConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_next_price(self, current_spot: float) -> float:
        dt = self.config.dt
        z = self.rng.standard_normal()
        drift = (self.config.mu - 0.5 * self.config.sigma**2) * dt
        diffusion = self.config.sigma * np.sqrt(dt) * z
        return float(current_spot * np.exp(drift + diffusion))

    def sample_path(self, n_steps: int | None = None) -> np.ndarray:
        steps = n_steps or self.config.steps
        path = np.empty(steps + 1, dtype=np.float64)
        path[0] = self.config.S0

        for t in range(steps):
            path[t + 1] = self.sample_next_price(path[t])

        return path
