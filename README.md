# Risk-Sensitive Derivative Hedging using Deep Learning + Reinforcement Learning

This repository now includes a practical foundation for training on:
1. **Synthetic GBM data** (fast prototyping), and
2. **Historical LSE data** (real training pipeline).

## What is already implemented

- RL environment for option hedging on simulated GBM paths (`DerivativeHedgingEnv`).
- RL environment for replaying historical prices (`HistoricalDerivativeHedgingEnv`).
- Black–Scholes utilities (price + delta).
- Data pipeline for LSE CSV preparation.
- PPO training entrypoint (`train.py`) and recurrent PPO (LSTM policy) entrypoint for LSE (`train_lse.py`).
- Baseline delta-hedging runner.

---

## From where to download LSE data? Any inbuilt libraries?

Short answer:
- **Python has no built-in standard-library API for LSE historical OHLC data.**
- Use external providers/libraries.

Recommended options:
1. **Yahoo Finance via `yfinance`** (easiest and already integrated in this repo)
   - LSE symbols are usually suffixed with `.L` (e.g., `VOD.L`, `BP.L`, `HSBA.L`).
2. **Kaggle datasets** (good for frozen/reproducible snapshots).
3. **Official/exchange/vendor feeds** (Refinitiv, Bloomberg, ICE, etc.) for institutional-grade data.

### Built-in downloader added in this repo

Use:

```bash
python scripts/download_lse_data.py --tickers VOD.L BP.L HSBA.L --start 2014-01-01 --end 2024-12-31 --output data/raw/lse_prices.csv
```

This creates a CSV with required schema:
- `date`
- `symbol`
- `close`

---

## Files required to start training on LSE data

You need these files in place:

1. **Raw historical data CSV**
   - Path: `data/raw/lse_prices.csv`
   - Required columns:
     - `date`
     - `symbol`
     - `close`

2. **Market config**
   - Path: `configs/market.yaml`
   - Defines option/market assumptions (strike, maturity steps, rate, transaction cost).

3. **Training config**
   - Path: `configs/training.yaml`
   - Defines optimizer/algorithm hyperparameters (`learning_rate`, `epochs`, `total_timesteps`, ...).

4. **Data config**
   - Path: `configs/data.yaml`
   - Defines CSV schema mapping + symbol selection + train split.

5. **Prepared feature file** (auto-generated)
   - Path: `data/processed/lse_features.csv`
   - Created by running `scripts/prepare_lse_data.py`.

---

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Current dependencies:
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `torch`
- `gymnasium`
- `stable-baselines3`
- `sb3-contrib` *(for RecurrentPPO / LSTM policy)*
- `pyyaml`
- `tqdm`
- `scikit-learn`
- `tensorboard`
- `jupyter`
- `ipykernel`
- `yfinance` *(download LSE data from Yahoo Finance)*

---

## Training workflows

### A) Synthetic-data baseline

```bash
python train.py
```

Output:
- `artifacts/ppo_hedging.zip`

### B) Historical LSE training (RL + DL hybrid via recurrent policy)

1. Download data (or place your own CSV) to `data/raw/lse_prices.csv`:

```bash
python scripts/download_lse_data.py --tickers VOD.L --start 2010-01-01 --end 2024-12-31
```

2. Build features:

```bash
python scripts/prepare_lse_data.py
```

3. Train recurrent PPO (LSTM policy):

```bash
python train_lse.py
```

Output:
- `artifacts/recurrent_ppo_lse.zip`

---

## Google Colab setup (recommended for longer training)

### 1. In Colab, clone/upload project to:
- `/content/Derivative-Hedging`

### 2. Install deps
```python
!pip install -r /content/Derivative-Hedging/requirements.txt
```

### 3. Download or upload LSE CSV
```python
!python /content/Derivative-Hedging/scripts/download_lse_data.py --tickers VOD.L BP.L --start 2010-01-01 --end 2024-12-31 --output /content/Derivative-Hedging/data/raw/lse_prices.csv
```

### 4. Prepare features + train
```python
!python /content/Derivative-Hedging/scripts/prepare_lse_data.py
!python /content/Derivative-Hedging/train_lse.py
```

### 5. Export trained model to Google Drive
Use:
- `colab/train_colab.py`

It mounts Drive and copies:
- `/content/Derivative-Hedging/artifacts/recurrent_ppo_lse.zip`

into:
- `/content/drive/MyDrive/derivative_hedging_exports/`

---


## Demo Frontend Dashboard

A lightweight demo dashboard is available in `frontend/` with:
- asset class selector: **Oil, Bullion, Forex, Stocks**
- LSE-oriented instrument options (e.g., `BP.L`, `SHEL.L`, `VOD.L`, `HSBA.L`)
- demo hedging metrics cards
- simple canvas line chart for normalized hedging error / PnL visualization

Run locally:

```bash
python -m http.server 8000
```

Then open:
- `http://localhost:8000/frontend/`

---
#### Results after training RL model

Derivative Hedging using RL — Final Results (NB7)

Dataset:    12 instruments (LSE equities, oil, gold, FX), 2020–2025
Model:      PPO with MLP policy (512→256→128), 13.7M total training steps
Baseline:   Black-Scholes delta hedging

Primary metric          RL Agent    BS Baseline    vs Target
─────────────────────────────────────────────────────────────
Bull market PnL         +0.00954    +0.00140       ✅ 6.8× BS
Sharpe ratio            +3.925      +0.883         ✅ > 0.37
avg |portfolio delta|    0.012       —             ✅ < 0.10
Win rate (bull)         51.7%        48.3%         ✅ > 50%
t-test p-value          0.032        —             ✅ < 0.05
Bear market win rate    100%         —             ✅
BuyU / SellU            1.3% / 12%   —             ✅ < 35% each
Close All frequency      0%          —             ✅ < 10%

Three degenerate strategies discovered and fixed iteratively:
  NB2→NB3: Close All exploit (57% → 0%)
  NB4→NB5: Buy-underlying momentum (75% → 2%)
  NB5→NB6: Sell-underlying bear momentum (88% → 12%)
