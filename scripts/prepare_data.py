"""
Data Preparation Script
========================

Downloads market data and generates training/validation/test tensors.

Usage:
    python scripts/prepare_data.py --universe sp500 --start 2010-01-01 --end 2023-12-31

Output files (in data/processed/):
    X_train.pt, y_train.pt  — Training set (feature tensors + forward returns)
    X_val.pt,   y_val.pt    — Validation set
    X_test.pt,  y_test.pt   — Test set
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="sp500", choices=["sp500", "custom"])
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--val-start", default="2021-01-01")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output", default="data/processed/")
    return parser.parse_args()


def build_dataset(
    market_data: dict,
    feature_engineer,
    seq_len: int,
    horizon: int,
) -> tuple:
    """Build (X, y) tensors from a dict of OHLCV DataFrames."""
    from neural_alpha.encoder.preprocessing import MarketFeatureEngineer

    X_all, y_all = [], []

    for ticker, df in market_data.items():
        try:
            features = feature_engineer.fit_transform(df)
            n = len(features)

            if n < seq_len + horizon:
                continue

            close = df["Close"].values[-n:]
            log_returns = np.log(close[1:] / close[:-1])

            for i in range(seq_len, n - horizon):
                window = features[i - seq_len: i]         # (seq_len, n_features)
                fwd_return = log_returns[i: i + horizon].sum()  # scalar

                X_all.append(window)
                y_all.append(fwd_return)

        except Exception as e:
            logger.warning(f"Skipping {ticker}: {e}")

    if not X_all:
        raise ValueError("No samples generated. Check data loading.")

    X = torch.tensor(np.stack(X_all), dtype=torch.float32)
    y = torch.tensor(np.array(y_all), dtype=torch.float32)
    return X, y


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from neural_alpha.utils.data_loader import MarketDataLoader
    from neural_alpha.encoder.preprocessing import MarketFeatureEngineer

    loader = MarketDataLoader(cache_dir="data/raw/")
    feature_engineer = MarketFeatureEngineer()

    # Determine ticker universe
    if args.universe == "sp500" and args.tickers is None:
        logger.info("Loading S&P 500 universe (top 100 liquid)...")
        market_data = loader.load_sp500_universe(start=args.start, end=args.end, n=100)
    else:
        tickers = args.tickers or ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        market_data = loader.load(tickers, start=args.start, end=args.end)

    logger.info(f"Loaded {len(market_data)} tickers.")

    # Build full dataset
    logger.info("Building feature tensors...")
    X, y = build_dataset(market_data, feature_engineer, args.seq_len, args.horizon)
    logger.info(f"Total samples: {len(X)}, X shape: {X.shape}")

    # Split by time (approximate — uses row counts as proxy)
    n = len(X)
    val_frac = 0.15
    test_frac = 0.15
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train: n_train + n_val], y[n_train: n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Save
    torch.save(X_train, output_dir / "X_train.pt")
    torch.save(y_train, output_dir / "y_train.pt")
    torch.save(X_val, output_dir / "X_val.pt")
    torch.save(y_val, output_dir / "y_val.pt")
    torch.save(X_test, output_dir / "X_test.pt")
    torch.save(y_test, output_dir / "y_test.pt")

    logger.info(f"Saved tensors to {output_dir}")


if __name__ == "__main__":
    main()
