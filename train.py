from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from hedging_rl.config import load_market_config, load_training_config
from hedging_rl.env import DerivativeHedgingEnv


def make_env(market_cfg_path: str):
    market_cfg = load_market_config(market_cfg_path)

    def _factory():
        env = DerivativeHedgingEnv(config=market_cfg)
        return Monitor(env)

    return _factory


def main() -> None:
    market_path = Path("configs/market.yaml")
    training_path = Path("configs/training.yaml")

    market_cfg = load_market_config(market_path)
    train_cfg = load_training_config(training_path)

    env = DummyVecEnv([make_env(str(market_path))])

    if train_cfg.algo.upper() != "PPO":
        raise ValueError(f"Unsupported algorithm '{train_cfg.algo}'. This scaffold currently supports PPO.")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=train_cfg.learning_rate,
        batch_size=train_cfg.batch_size,
        n_epochs=train_cfg.epochs,
        gamma=train_cfg.gamma,
        verbose=1,
    )

    model.learn(total_timesteps=train_cfg.total_timesteps)

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    model.save(output_dir / "ppo_hedging")

    print("Training complete. Saved model to artifacts/ppo_hedging.zip")
    print(f"Market setup: S0={market_cfg.S0}, K={market_cfg.K}, sigma={market_cfg.sigma}, steps={market_cfg.steps}")


if __name__ == "__main__":
    main()
