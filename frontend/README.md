# RL Delta Hedging Dashboard

Interactive Streamlit dashboard for the NB9 variance-minimizing RL hedging model.

## Run

```bash
cd Derivative-Hedging-using-RL-DL
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Opens at http://localhost:8501

## Features

- **Market Setup** — set spot price, strike, vol, maturity, risk-free rate for any instrument
- **Price Path Generator** — 5 regimes: GBM, Trending Bull/Bear, Mean Reverting, Vol Spike, Crash
- **Live Option Pricer** — real-time B-S call/put/delta/gamma/vega/theta at current params
- **6 charts** — price+actions, portfolio delta, returns vs benchmarks, positions, Greeks
- **Trade log** — every action the agent took with price, delta, portfolio value
- **Episode summary** — hedging error, premium collected, cash flow, return comparison

## What the RL model does

NB9 uses a variance-minimizing reward: `reward = -(step_pnl)²`
Target is 0% return with minimum variance — not profit maximization.
Strategy: sell ATM call → collect theta → delta-hedge with underlying.
Hedging error: 0.00006 vs BS baseline 0.00137 (99.6% variance reduction).
