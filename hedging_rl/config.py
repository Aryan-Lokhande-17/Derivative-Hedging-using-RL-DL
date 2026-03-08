from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class MarketConfig:
    mu: float
    sigma: float
    r: float
    S0: float
    K: float
    T: float
    steps: int
    transaction_cost: float

    @property
    def dt(self) -> float:
        return self.T / self.steps


@dataclass(slots=True)
class TrainingConfig:
    algo: str = "PPO"
    learning_rate: float = 3e-4
    batch_size: int = 2048
    epochs: int = 10
    gamma: float = 0.99
    cvar_alpha: float = 0.95
    total_timesteps: int = 250_000


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_market_config(path: str | Path) -> MarketConfig:
    return MarketConfig(**load_yaml(path))


def load_training_config(path: str | Path) -> TrainingConfig:
    data = load_yaml(path)
    return TrainingConfig(**data)
