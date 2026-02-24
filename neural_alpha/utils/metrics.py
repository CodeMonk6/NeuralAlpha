"""
Backtest Metrics
================

Comprehensive evaluation metrics for alpha signal quality and
strategy performance.

Metrics:
    Signal quality:
        IC (Information Coefficient), ICIR, Hit Rate, Long-Short Spread

    Portfolio performance:
        Sharpe, Sortino, Calmar, Max Drawdown, Annualized Return,
        Annualized Volatility, Information Ratio, Turnover
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class BacktestMetrics:
    """
    Compute portfolio and signal quality metrics.

    Args:
        risk_free_rate: Annual risk-free rate for Sharpe computation (default: 4.5%).
        trading_days:   Number of trading days per year (default: 252).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        trading_days: int = 252,
    ):
        self.rfr = risk_free_rate
        self.T = trading_days

    # ------------------------------------------------------------------
    # Signal Quality Metrics
    # ------------------------------------------------------------------

    def information_coefficient(
        self,
        alpha_scores: pd.Series,
        forward_returns: pd.Series,
    ) -> float:
        """
        Rank correlation (Spearman) between alpha scores and forward returns.
        IC > 0.05 is considered economically meaningful.
        IC > 0.10 is strong.
        """
        from scipy.stats import spearmanr
        mask = alpha_scores.notna() & forward_returns.notna()
        ic, _ = spearmanr(alpha_scores[mask], forward_returns[mask])
        return float(ic)

    def ic_information_ratio(
        self,
        ic_series: pd.Series,
    ) -> float:
        """
        ICIR = mean(IC) / std(IC). 
        Measures consistency of alpha. ICIR > 0.5 is excellent.
        """
        return float(ic_series.mean() / ic_series.std().clip(lower=1e-9))

    def hit_rate(
        self,
        alpha_scores: pd.Series,
        forward_returns: pd.Series,
    ) -> float:
        """Fraction of signals where direction (sign) of alpha matched sign of return."""
        correct = (np.sign(alpha_scores) == np.sign(forward_returns)).sum()
        total = len(alpha_scores.dropna())
        return float(correct / max(total, 1))

    def long_short_spread(
        self,
        alpha_scores: pd.Series,
        forward_returns: pd.Series,
        n_quantiles: int = 5,
    ) -> float:
        """
        Return spread between top-quintile and bottom-quintile assets.
        Higher is better.
        """
        quantile_labels = pd.qcut(alpha_scores, q=n_quantiles, labels=False, duplicates="drop")
        top = forward_returns[quantile_labels == n_quantiles - 1].mean()
        bottom = forward_returns[quantile_labels == 0].mean()
        return float(top - bottom)

    # ------------------------------------------------------------------
    # Portfolio Performance Metrics
    # ------------------------------------------------------------------

    def annualized_return(self, daily_returns: pd.Series) -> float:
        """Compound annualized return."""
        cumulative = (1 + daily_returns).prod()
        n = len(daily_returns)
        return float(cumulative ** (self.T / max(n, 1)) - 1)

    def annualized_volatility(self, daily_returns: pd.Series) -> float:
        """Annualized standard deviation of daily returns."""
        return float(daily_returns.std() * np.sqrt(self.T))

    def sharpe_ratio(self, daily_returns: pd.Series) -> float:
        """Annualized Sharpe ratio."""
        excess = daily_returns - self.rfr / self.T
        vol = daily_returns.std() * np.sqrt(self.T)
        return float(excess.mean() * self.T / max(vol, 1e-9))

    def sortino_ratio(self, daily_returns: pd.Series) -> float:
        """Annualized Sortino ratio (uses downside deviation)."""
        excess = daily_returns - self.rfr / self.T
        downside = daily_returns[daily_returns < 0].std() * np.sqrt(self.T)
        return float(excess.mean() * self.T / max(downside, 1e-9))

    def max_drawdown(self, daily_returns: pd.Series) -> float:
        """Maximum peak-to-trough drawdown (negative number)."""
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        return float(drawdown.min())

    def calmar_ratio(self, daily_returns: pd.Series) -> float:
        """Annualized return / abs(max drawdown)."""
        ann_ret = self.annualized_return(daily_returns)
        mdd = abs(self.max_drawdown(daily_returns))
        return float(ann_ret / max(mdd, 1e-9))

    def information_ratio(
        self,
        daily_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Information Ratio vs benchmark."""
        active = daily_returns - benchmark_returns
        tracking_error = active.std() * np.sqrt(self.T)
        return float(active.mean() * self.T / max(tracking_error, 1e-9))

    def turnover(
        self,
        weights_df: pd.DataFrame,
    ) -> float:
        """
        Average one-way portfolio turnover per rebalance.

        Args:
            weights_df: (T, n_assets) DataFrame of portfolio weights.

        Returns:
            avg_one_way_turnover: Scalar.
        """
        diffs = weights_df.diff().abs().sum(axis=1)
        return float(diffs.mean())

    # ------------------------------------------------------------------
    # Full Report
    # ------------------------------------------------------------------

    def full_report(
        self,
        daily_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        alpha_scores: Optional[pd.Series] = None,
        forward_returns: Optional[pd.Series] = None,
    ) -> dict:
        """
        Generate a comprehensive performance report.

        Returns a dict with all metrics. Print with .print_report().
        """
        report = {
            "Annualized Return":    self.annualized_return(daily_returns),
            "Annualized Volatility": self.annualized_volatility(daily_returns),
            "Sharpe Ratio":         self.sharpe_ratio(daily_returns),
            "Sortino Ratio":        self.sortino_ratio(daily_returns),
            "Max Drawdown":         self.max_drawdown(daily_returns),
            "Calmar Ratio":         self.calmar_ratio(daily_returns),
        }

        if benchmark_returns is not None:
            report["Information Ratio"] = self.information_ratio(daily_returns, benchmark_returns)
            report["Benchmark Annualized Return"] = self.annualized_return(benchmark_returns)
            report["Excess Return"] = report["Annualized Return"] - report["Benchmark Annualized Return"]

        if alpha_scores is not None and forward_returns is not None:
            report["IC"] = self.information_coefficient(alpha_scores, forward_returns)
            report["Hit Rate"] = self.hit_rate(alpha_scores, forward_returns)
            report["Long-Short Spread"] = self.long_short_spread(alpha_scores, forward_returns)

        return report

    def print_report(self, report: dict) -> None:
        """Pretty-print a performance report dict."""
        print("═" * 52)
        print("        NeuralAlpha Performance Report")
        print("═" * 52)
        for metric, value in report.items():
            if isinstance(value, float):
                if "Return" in metric or "Drawdown" in metric or "Spread" in metric:
                    print(f"  {metric:<30s}  {value:>+8.2%}")
                else:
                    print(f"  {metric:<30s}  {value:>8.4f}")
        print("═" * 52)
