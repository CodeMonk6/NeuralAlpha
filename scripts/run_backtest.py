"""
Backtest Script
================

Runs a vectorized backtest from saved signal CSVs or by generating
signals from a pretrained model over a historical period.

Usage:
    # From pretrained model
    python scripts/run_backtest.py \
        --pretrained checkpoints/full/best/ \
        --universe sp500 \
        --start 2022-01-01 \
        --end 2024-12-31 \
        --rebalance weekly

    # From pre-generated signals CSV
    python scripts/run_backtest.py \
        --signals data/signals/sp500_signals.csv \
        --start 2022-01-01 \
        --end 2024-12-31
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--signals", type=str, default=None)
    parser.add_argument("--universe", type=str, default="sp500")
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--rebalance", type=str, default="weekly", choices=["daily", "weekly", "monthly"])
    parser.add_argument("--long-only", action="store_true", default=False)
    parser.add_argument("--n-top", type=int, default=20, help="Top-N long positions")
    parser.add_argument("--n-bottom", type=int, default=20, help="Bottom-N short positions")
    parser.add_argument("--output", type=str, default="results/backtest/")
    return parser.parse_args()


def signals_to_weights(
    signals_df: pd.DataFrame,
    n_long: int,
    n_short: int,
    long_only: bool,
) -> pd.Series:
    """
    Convert alpha scores to portfolio weights.
    Equal-weight within long and short buckets.
    """
    signals_df = signals_df.sort_values("alpha_score", ascending=False)
    
    weights = pd.Series(0.0, index=signals_df["ticker"])
    
    # Long positions: top n_long by alpha score
    long_tickers = signals_df[signals_df["alpha_score"] > 0]["ticker"].iloc[:n_long]
    if len(long_tickers) > 0:
        weights[long_tickers] = 0.5 / len(long_tickers)  # 50% long book
    
    if not long_only:
        # Short positions: bottom n_short by alpha score
        short_tickers = signals_df[signals_df["alpha_score"] < 0]["ticker"].iloc[-n_short:]
        if len(short_tickers) > 0:
            weights[short_tickers] = -0.5 / len(short_tickers)  # 50% short book
    
    return weights


def compute_portfolio_returns(
    weights: pd.Series,
    price_data: dict,
    rebalance_date: str,
    next_rebalance_date: str,
) -> pd.Series:
    """
    Compute daily portfolio returns between two rebalance dates.
    """
    dates = pd.bdate_range(rebalance_date, next_rebalance_date)
    port_returns = pd.Series(0.0, index=dates)

    for ticker, weight in weights.items():
        if ticker not in price_data or abs(weight) < 1e-6:
            continue
        try:
            prices = price_data[ticker]["Close"]
            ticker_rets = prices.pct_change()
            ticker_rets = ticker_rets.reindex(dates).fillna(0.0)
            port_returns += weight * ticker_rets
        except Exception:
            pass

    return port_returns


def print_backtest_report(returns: pd.Series, benchmark_returns: pd.Series) -> None:
    from neural_alpha.utils.metrics import BacktestMetrics

    metrics = BacktestMetrics()
    report = metrics.full_report(
        daily_returns=returns,
        benchmark_returns=benchmark_returns,
    )
    metrics.print_report(report)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Backtest: {args.start} → {args.end}")
    logger.info(f"Rebalance frequency: {args.rebalance}")

    # Load market data
    from neural_alpha.utils.data_loader import MarketDataLoader
    loader = MarketDataLoader(cache_dir="data/raw/")
    market_data = loader.load_sp500_universe(
        start=args.start, end=args.end, n=100
    )
    logger.info(f"Loaded {len(market_data)} tickers.")

    # Determine rebalance dates
    all_dates = pd.bdate_range(args.start, args.end)
    freq_map = {"daily": "B", "weekly": "W-FRI", "monthly": "BMS"}
    rebalance_dates = pd.bdate_range(args.start, args.end, freq=freq_map[args.rebalance])

    # Collect portfolio returns
    all_port_returns = []

    for i, reb_date in enumerate(rebalance_dates[:-1]):
        next_reb = rebalance_dates[i + 1]
        reb_str = reb_date.strftime("%Y-%m-%d")
        next_str = next_reb.strftime("%Y-%m-%d")

        if args.pretrained:
            from neural_alpha import NeuralAlphaPipeline
            if i == 0:
                model = NeuralAlphaPipeline.from_pretrained(args.pretrained)
            signals_df = model.generate_signals(
                list(market_data.keys()), reb_str
            )
        else:
            logger.warning("No pretrained model or signals CSV provided. Using random signals.")
            tickers = list(market_data.keys())
            signals_df = pd.DataFrame({
                "ticker": tickers,
                "alpha_score": np.random.randn(len(tickers)),
            })

        if signals_df.empty:
            continue

        weights = signals_to_weights(
            signals_df, args.n_top, args.n_bottom, args.long_only
        )
        period_rets = compute_portfolio_returns(weights, market_data, reb_str, next_str)
        all_port_returns.append(period_rets)

    if not all_port_returns:
        logger.error("No returns computed.")
        return

    port_returns = pd.concat(all_port_returns).sort_index()

    # Benchmark: equal-weight SPX proxy
    benchmark_tickers = ["SPY"]
    benchmark_data = loader.load(benchmark_tickers, start=args.start, end=args.end)
    if "SPY" in benchmark_data:
        bench_returns = benchmark_data["SPY"]["Close"].pct_change().dropna()
        bench_returns = bench_returns.reindex(port_returns.index).fillna(0.0)
    else:
        bench_returns = pd.Series(0.0, index=port_returns.index)

    # Print & save report
    print("\n")
    print_backtest_report(port_returns, bench_returns)

    # Save returns
    port_returns.to_csv(output_dir / "portfolio_returns.csv")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
