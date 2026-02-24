"""
Market Data Loader
==================

Downloads and caches historical OHLCV data for a universe of tickers.
Uses yfinance as the data backend with local disk caching.

Usage:
    loader = MarketDataLoader(cache_dir="data/raw/")
    data = loader.load(["AAPL", "MSFT"], start="2020-01-01", end="2024-01-01")
    # Returns dict[ticker → pd.DataFrame]
"""

import os
import pickle
import hashlib
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """
    Downloads, caches, and validates OHLCV market data.

    Args:
        cache_dir:    Directory to store cached data files.
        use_cache:    Whether to use cached data (disable for live signals).
        align_dates:  If True, align all tickers to common date range
                      (fills missing dates with forward-fill).
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(
        self,
        cache_dir: str = "data/raw/",
        use_cache: bool = True,
        align_dates: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.align_dates = align_dates
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        tickers: List[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for a list of tickers.

        Args:
            tickers:  List of ticker symbols (e.g., ["AAPL", "MSFT"]).
            start:    Start date string "YYYY-MM-DD".
            end:      End date string "YYYY-MM-DD".
            interval: Data frequency ("1d", "1wk", "1mo").

        Returns:
            data: Dict mapping ticker → pd.DataFrame with OHLCV columns.
        """
        data = {}
        failed = []

        for ticker in tickers:
            try:
                df = self._load_single(ticker, start, end, interval)
                if df is not None and len(df) > 0:
                    data[ticker] = df
                else:
                    logger.warning(f"No data returned for {ticker}")
                    failed.append(ticker)
            except Exception as e:
                logger.error(f"Failed to load {ticker}: {e}")
                failed.append(ticker)

        if failed:
            logger.warning(f"Failed tickers ({len(failed)}): {failed}")

        if self.align_dates and len(data) > 1:
            data = self._align_dates(data)

        logger.info(f"Loaded data for {len(data)}/{len(tickers)} tickers.")
        return data

    def _load_single(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Load data for a single ticker, using cache if available."""
        cache_key = self._cache_key(ticker, start, end, interval)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        if self.use_cache and cache_path.exists():
            logger.debug(f"Loading {ticker} from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Download from yfinance
        try:
            import yfinance as yf
            logger.info(f"Downloading {ticker} [{start} → {end}]")
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

            if df.empty:
                return None

            # Standardize column names
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df[self.REQUIRED_COLUMNS].copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Cache to disk
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)

            return df

        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

    def _align_dates(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Align all DataFrames to a common date range."""
        # Find common date range
        all_dates = sorted(
            set.intersection(*[set(df.index) for df in data.values()])
        )
        if not all_dates:
            logger.warning("No common dates found. Returning unaligned data.")
            return data

        aligned = {}
        for ticker, df in data.items():
            aligned[ticker] = df.loc[df.index.isin(all_dates)].copy()

        return aligned

    def load_sp500_universe(
        self, start: str, end: str, n: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Convenience: load a subset of S&P 500 tickers.
        Uses a hardcoded list of liquid large-caps.
        """
        sp500_liquid = [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY",
            "AVGO", "TSLA", "JPM", "V", "UNH", "XOM", "MA", "PG", "JNJ",
            "HD", "MRK", "ABBV", "COST", "CVX", "CRM", "BAC", "NFLX",
            "AMD", "ORCL", "PEP", "KO", "TMO", "WMT", "CSCO", "ACN", "MCD",
            "ABT", "DHR", "LIN", "TXN", "PM", "ADBE", "NOW", "NEE", "AMGN",
            "UNP", "MS", "IBM", "GS", "QCOM", "INTU", "ISRG",
        ]
        tickers = sp500_liquid[:n]
        return self.load(tickers, start, end)

    @staticmethod
    def _cache_key(ticker: str, start: str, end: str, interval: str) -> str:
        raw = f"{ticker}_{start}_{end}_{interval}"
        return hashlib.md5(raw.encode()).hexdigest()[:12] + f"_{ticker}"

    def validate(self, data: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
        """
        Run data quality checks on loaded data.

        Returns:
            report: Dict[ticker → quality_metrics]
        """
        report = {}
        for ticker, df in data.items():
            n_rows = len(df)
            n_nulls = df.isnull().sum().sum()
            pct_null = n_nulls / (n_rows * len(df.columns)) * 100
            has_neg_prices = (df[["Open", "High", "Low", "Close"]] <= 0).any().any()
            report[ticker] = {
                "n_rows": n_rows,
                "pct_null": round(pct_null, 2),
                "has_negative_prices": bool(has_neg_prices),
                "date_range": f"{df.index.min().date()} → {df.index.max().date()}",
            }
        return report
