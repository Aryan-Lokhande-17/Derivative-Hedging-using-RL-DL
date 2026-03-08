from __future__ import annotations

from hedging_rl.data import build_lse_features


if __name__ == "__main__":
    df = build_lse_features("configs/data.yaml")
    print(f"Saved processed dataset with {len(df)} rows to configs/data.yaml -> output_csv")
