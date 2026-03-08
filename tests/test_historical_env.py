import numpy as np
import pandas as pd

from hedging_rl.env_historical import HistoricalDerivativeHedgingEnv


def test_historical_env_shapes():
    prices = pd.Series(np.linspace(90, 110, 300))
    env = HistoricalDerivativeHedgingEnv(
        prices=prices,
        strike=100,
        rate=0.01,
        maturity_steps=30,
        transaction_cost=0.001,
    )

    obs, _ = env.reset(seed=123)
    assert obs.shape == (4,)

    next_obs, reward, terminated, truncated, info = env.step(np.array([0.2], dtype=np.float32))
    assert next_obs.shape == (4,)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert "replication_error" in info
