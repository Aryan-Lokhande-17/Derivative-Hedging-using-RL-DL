import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from hedging_rl.config import load_market_config
from hedging_rl.env import DerivativeHedgingEnv


market_cfg = load_market_config("configs/market.yaml")

env = DerivativeHedgingEnv(config=market_cfg)


model = PPO.load("artifacts/ppo_hedging.zip")


obs, _ = env.reset()

prices = []
rl_positions = []
bs_positions = []
pnl_rl = []
pnl_bs = []

done = False

while not done:

    price = obs[2]
    prices.append(price)

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    rl_position = obs[3]
    bs_position = obs[4]

    rl_positions.append(rl_position)
    bs_positions.append(bs_position)

    pnl_rl.append(reward)

    if len(prices) > 1:
        dS = prices[-1] - prices[-2]

        if len(pnl_bs) == 0:
            pnl_bs.append(bs_position * dS)
        else:
            pnl_bs.append(pnl_bs[-1] + bs_position * dS)
    else:
        pnl_bs.append(0)

    done = terminated or truncated

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