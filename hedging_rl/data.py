from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class DataConfigError(ValueError):
    pass


def load_data_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    required = {
        "input_csv",
        "output_csv",
        "symbol_column",
        "date_column",
        "price_column",
        "symbol",
        "lookback_window",
        "train_split",
    }
    missing = required - set(data)
    if missing:
        raise DataConfigError(f"Missing keys in data config: {sorted(missing)}")
    return data


def build_lse_features(config_path: str | Path = "configs/data.yaml") -> pd.DataFrame:
    cfg = load_data_config(config_path)
    raw = pd.read_csv(cfg["input_csv"])

    df = raw.loc[raw[cfg["symbol_column"]] == cfg["symbol"]].copy()
    if df.empty:
        raise DataConfigError(f"No rows found for symbol {cfg['symbol']}")

    df[cfg["date_column"]] = pd.to_datetime(df[cfg["date_column"]])
    df = df.sort_values(cfg["date_column"]).reset_index(drop=True)

    close = df[cfg["price_column"]].astype(float)
    df["log_return"] = np.log(close / close.shift(1))
    df["realized_vol_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)
    df["normalized_price"] = close / close.iloc[0]
    df = df.dropna().reset_index(drop=True)

    output_path = Path(cfg["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
