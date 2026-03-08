from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_lse_data(tickers: list[str], start: str, end: str, output_csv: str) -> Path:
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        hist = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if hist.empty:
            continue

        frame = hist.reset_index()[["Date", "Close"]].copy()
        frame.columns = ["date", "close"]
        frame["symbol"] = ticker
        frames.append(frame[["date", "symbol", "close"]])

    if not frames:
        raise ValueError("No data downloaded. Check ticker symbols, date range, or network connectivity.")

    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LSE historical prices (Yahoo Finance .L tickers).")
    parser.add_argument("--tickers", nargs="+", default=["VOD.L"], help="LSE tickers, e.g. VOD.L BP.L HSBA.L")
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="data/raw/lse_prices.csv", help="Output CSV path")
    args = parser.parse_args()

    output = download_lse_data(args.tickers, args.start, args.end, args.output)
    print(f"Saved LSE dataset to {output}")


if __name__ == "__main__":
    main()
