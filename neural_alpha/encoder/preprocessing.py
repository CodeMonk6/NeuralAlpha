"""
Market Feature Engineering
===========================

Transforms raw OHLCV data + optional alternative data into a rich
feature matrix suitable for the MarketEncoder.

Features produced (32 total by default):
    Price-derived (8):
        log_return_1d, log_return_5d, log_return_21d
        high_low_range, close_to_open
        rolling_zscore_21d, rolling_zscore_63d
        price_vs_52w_high
    Volume (4):
        log_volume, volume_zscore_21d, dollar_volume_rank, turnover
    Technical indicators (12):
        rsi_14, macd, macd_signal, macd_hist
        bb_upper, bb_lower, bb_width
        atr_14, adx_14
        ema_9, ema_21, ema_ratio
    Fundamental / macro (4):
        earnings_revision_30d, analyst_sentiment
        sector_relative_strength, market_beta_63d
    Calendar (4):
        day_of_week_sin, day_of_week_cos
        month_sin, month_cos
"""

import numpy as np
import pandas as pd
from typing import Optional


class MarketFeatureEngineer:
    """
    Transforms a raw OHLCV DataFrame into a feature matrix.

    Args:
        include_fundamentals: Whether to include fundamental/macro features.
                              Set False for pure price/volume models.
        zscore_window:        Rolling window for z-score normalization.
    """

    FEATURE_NAMES = [
        # Price-derived
        "log_return_1d", "log_return_5d", "log_return_21d",
        "high_low_range", "close_to_open",
        "rolling_zscore_21d", "rolling_zscore_63d", "price_vs_52w_high",
        # Volume
        "log_volume", "volume_zscore_21d", "dollar_volume_rank", "turnover",
        # Technical
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width",
        "atr_14", "adx_14", "ema_9", "ema_21", "ema_ratio",
        # Fundamental / macro
        "earnings_revision_30d", "analyst_sentiment",
        "sector_relative_strength", "market_beta_63d",
        # Calendar
        "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
    ]

    def __init__(self, include_fundamentals: bool = True, zscore_window: int = 252):
        self.include_fundamentals = include_fundamentals
        self.zscore_window = zscore_window

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit (compute rolling stats) and transform a raw OHLCV DataFrame.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                and a DatetimeIndex. Optionally includes:
                ['earnings_revision', 'analyst_rating', 'sector_rs', 'market_return']

        Returns:
            features: np.ndarray of shape (T, n_features), with NaN rows at start dropped.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        feats = {}

        close = df["close"]
        log_close = np.log(close.clip(lower=1e-9))

        # --- Price-derived ---
        feats["log_return_1d"] = log_close.diff(1)
        feats["log_return_5d"] = log_close.diff(5)
        feats["log_return_21d"] = log_close.diff(21)
        feats["high_low_range"] = (df["high"] - df["low"]) / close
        feats["close_to_open"] = (close - df["open"]) / df["open"].clip(lower=1e-9)
        feats["rolling_zscore_21d"] = self._rolling_zscore(close, 21)
        feats["rolling_zscore_63d"] = self._rolling_zscore(close, 63)
        feats["price_vs_52w_high"] = close / close.rolling(252).max().clip(lower=1e-9) - 1

        # --- Volume ---
        vol = df["volume"].clip(lower=1)
        feats["log_volume"] = np.log(vol)
        feats["volume_zscore_21d"] = self._rolling_zscore(vol, 21)
        feats["dollar_volume_rank"] = (close * vol).rank(pct=True)
        feats["turnover"] = vol / vol.rolling(21).mean().clip(lower=1e-9)

        # --- Technical Indicators ---
        feats["rsi_14"] = self._rsi(close, 14) / 100.0  # normalize 0–1
        macd_line, signal_line = self._macd(close)
        feats["macd"] = macd_line / close.clip(lower=1e-9)
        feats["macd_signal"] = signal_line / close.clip(lower=1e-9)
        feats["macd_hist"] = (macd_line - signal_line) / close.clip(lower=1e-9)
        bb_up, bb_lo = self._bollinger_bands(close)
        bb_width = (bb_up - bb_lo) / close.clip(lower=1e-9)
        feats["bb_upper"] = (close - bb_up) / close.clip(lower=1e-9)
        feats["bb_lower"] = (close - bb_lo) / close.clip(lower=1e-9)
        feats["bb_width"] = bb_width
        feats["atr_14"] = self._atr(df, 14) / close.clip(lower=1e-9)
        feats["adx_14"] = self._adx(df, 14) / 100.0
        ema9 = close.ewm(span=9).mean()
        ema21 = close.ewm(span=21).mean()
        feats["ema_9"] = (close - ema9) / close.clip(lower=1e-9)
        feats["ema_21"] = (close - ema21) / close.clip(lower=1e-9)
        feats["ema_ratio"] = ema9 / ema21.clip(lower=1e-9) - 1

        # --- Fundamental / Macro ---
        if self.include_fundamentals and "earnings_revision" in df.columns:
            feats["earnings_revision_30d"] = df["earnings_revision"].fillna(0)
            feats["analyst_sentiment"] = df.get("analyst_rating", pd.Series(0, index=df.index)).fillna(0)
            feats["sector_relative_strength"] = df.get("sector_rs", pd.Series(0, index=df.index)).fillna(0)
            feats["market_beta_63d"] = self._rolling_beta(close, df.get("market_return"), 63)
        else:
            for k in ["earnings_revision_30d", "analyst_sentiment", "sector_relative_strength", "market_beta_63d"]:
                feats[k] = pd.Series(0.0, index=df.index)

        # --- Calendar ---
        idx = df.index
        feats["day_of_week_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
        feats["day_of_week_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)
        feats["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        feats["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

        # Assemble into DataFrame
        feat_df = pd.DataFrame(feats, index=df.index)[self.FEATURE_NAMES]

        # Drop rows with any NaN (warm-up period)
        feat_df = feat_df.dropna()

        return feat_df.values.astype(np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std().clip(lower=1e-9)
        return (series - mu) / sigma

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean().clip(lower=1e-9)
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    @staticmethod
    def _bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0):
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        return ma + n_std * std, ma - n_std * std

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        dm_plus = (high - prev_high).clip(lower=0)
        dm_minus = (prev_low - low).clip(lower=0)
        tr = MarketFeatureEngineer._atr(df, period)
        di_plus = 100 * dm_plus.rolling(period).mean() / tr.clip(lower=1e-9)
        di_minus = 100 * dm_minus.rolling(period).mean() / tr.clip(lower=1e-9)
        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).clip(lower=1e-9))
        return dx.rolling(period).mean()

    @staticmethod
    def _rolling_beta(asset: pd.Series, market: Optional[pd.Series], window: int) -> pd.Series:
        if market is None:
            return pd.Series(1.0, index=asset.index)
        r_a = np.log(asset.clip(lower=1e-9)).diff()
        r_m = np.log(market.clip(lower=1e-9)).diff()
        cov = r_a.rolling(window).cov(r_m)
        var = r_m.rolling(window).var().clip(lower=1e-9)
        return cov / var
