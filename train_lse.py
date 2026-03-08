from __future__ import annotations

from pathlib import Path

import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from hedging_rl.config import load_market_config, load_training_config
from hedging_rl.data import build_lse_features, load_data_config
from hedging_rl.env_historical import HistoricalDerivativeHedgingEnv


def _make_env(prices: pd.Series, strike: float, rate: float, maturity_steps: int, transaction_cost: float):
    def _factory():
        env = HistoricalDerivativeHedgingEnv(
            prices=prices,
            strike=strike,
            rate=rate,
            maturity_steps=maturity_steps,
            transaction_cost=transaction_cost,
        )
        return Monitor(env)

    return _factory


def main() -> None:
    market_cfg = load_market_config("configs/market.yaml")
    train_cfg = load_training_config("configs/training.yaml")
    data_cfg = load_data_config("configs/data.yaml")

    processed_path = Path(data_cfg["output_csv"])
    if not processed_path.exists():
        build_lse_features("configs/data.yaml")

    df = pd.read_csv(processed_path)
    prices = df[data_cfg["price_column"]].astype(float)

    split_idx = int(len(prices) * float(data_cfg["train_split"]))
    train_prices = prices.iloc[:split_idx].reset_index(drop=True)

    env = DummyVecEnv(
        [
            _make_env(
                prices=train_prices,
                strike=market_cfg.K,
                rate=market_cfg.r,
                maturity_steps=market_cfg.steps,
                transaction_cost=market_cfg.transaction_cost,
            )
        ]
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=train_cfg.learning_rate,
        batch_size=min(train_cfg.batch_size, 512),
        n_epochs=train_cfg.epochs,
        gamma=train_cfg.gamma,
        verbose=1,
    )
    model.learn(total_timesteps=train_cfg.total_timesteps)

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "recurrent_ppo_lse"
    model.save(out_path)
    print(f"Saved trained model to {out_path}.zip")


if __name__ == "__main__":
    main()
