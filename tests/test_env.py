import numpy as np

from hedging_rl.config import MarketConfig
from hedging_rl.env import DerivativeHedgingEnv


def test_env_reset_and_step_shapes():
    config = MarketConfig(mu=0.05, sigma=0.2, r=0.01, S0=100, K=100, T=1.0, steps=10, transaction_cost=0.001)
    env = DerivativeHedgingEnv(config=config, seed=42)

    obs, _ = env.reset()
    assert obs.shape == (5,)

    next_obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    assert next_obs.shape == (5,)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert "replication_error" in info
