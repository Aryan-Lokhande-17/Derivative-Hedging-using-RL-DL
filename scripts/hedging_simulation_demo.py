import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from hedging_rl.config import load_market_config
from hedging_rl.env import DerivativeHedgingEnv


# -----------------------------
# LOAD ENVIRONMENT
# -----------------------------

market_cfg = load_market_config("configs/market.yaml")

env = DerivativeHedgingEnv(config=market_cfg)

# -----------------------------
# LOAD TRAINED RL MODEL
# -----------------------------

model = PPO.load("artifacts/ppo_hedging.zip")


# -----------------------------
# RUN ONE EPISODE
# -----------------------------

obs, _ = env.reset()

prices = []
rl_positions = []
bs_positions = []
pnl_rl = []
pnl_bs = []

done = False

while not done:

    price = env.S
    prices.append(price)

    # ---------------------
    # RL HEDGE
    # ---------------------

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    rl_positions.append(env.position)
    pnl_rl.append(env.pnl)

    # ---------------------
    # BLACK SCHOLES HEDGE
    # ---------------------

    bs_positions.append(info["bs_delta"])
    pnl_bs.append(info["bs_pnl"])

    done = terminated or truncated


# -----------------------------
# PLOTS
# -----------------------------

plt.figure()

plt.plot(prices)

plt.title("GBM Price Path")
plt.xlabel("Time")
plt.ylabel("Price")

plt.show()


plt.figure()

plt.plot(rl_positions, label="RL Hedge")
plt.plot(bs_positions, label="Black-Scholes Hedge")

plt.title("Hedge Position Comparison")
plt.xlabel("Time")
plt.ylabel("Position")

plt.legend()

plt.show()


plt.figure()

plt.plot(pnl_rl, label="RL PnL")
plt.plot(pnl_bs, label="Black-Scholes PnL")

plt.title("PnL Comparison")
plt.xlabel("Time")
plt.ylabel("PnL")

plt.legend()

plt.show()


print("Final RL PnL:", pnl_rl[-1])
print("Final BS PnL:", pnl_bs[-1])