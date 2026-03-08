from pathlib import Path

import pandas as pd

from hedging_rl.data import build_lse_features


def test_build_lse_features(tmp_path: Path):
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "features.csv"
    cfg_path = tmp_path / "data.yaml"

    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    raw = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["VOD.L"] * 30,
            "close": [100 + i for i in range(30)],
        }
    )
    raw.to_csv(raw_path, index=False)

    cfg_path.write_text(
        "\n".join(
            [
                f"input_csv: {raw_path}",
                f"output_csv: {out_path}",
                "symbol_column: symbol",
                "date_column: date",
                "price_column: close",
                "symbol: VOD.L",
                "lookback_window: 20",
                "train_split: 0.8",
            ]
        )
    )

    df = build_lse_features(cfg_path)
    assert not df.empty
    assert out_path.exists()
    assert {"log_return", "realized_vol_20", "normalized_price"}.issubset(set(df.columns))
