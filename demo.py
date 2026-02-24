"""
NeuralAlpha Demo
================

Quick demonstration of signal generation for a handful of tickers.

Usage:
    python demo.py
    python demo.py --tickers AAPL MSFT NVDA TSLA --date 2024-06-01
    python demo.py --pretrained checkpoints/full/best/
"""

import argparse
import torch
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="NeuralAlpha Signal Demo")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "JPM", "XOM"],
        help="List of ticker symbols",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Signal date YYYY-MM-DD (defaults to today)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained checkpoint directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def demo_without_pretrained(tickers, date):
    """
    Demo mode: randomly initialized model for architecture demonstration.
    Not a trained model — signals are random.
    """
    from neural_alpha import NeuralAlphaPipeline

    console.print("\n[bold yellow]⚠ Running with randomly initialized weights (no pretrained checkpoint).[/bold yellow]")
    console.print("[yellow]Download pretrained weights with: python scripts/download_pretrained.py[/yellow]\n")

    model = NeuralAlphaPipeline(device="cpu")
    model.eval()

    # Generate synthetic demo signals
    import numpy as np
    np.random.seed(42)
    signals = []
    for ticker in tickers:
        alpha = np.random.uniform(-1, 1)
        confidence = np.random.uniform(0.4, 0.95)
        magnitude = np.random.uniform(0.1, 0.8)
        if confidence < 0.5:
            position = "NEUTRAL"
        elif alpha > 0.15:
            position = "LONG"
        elif alpha < -0.15:
            position = "SHORT"
        else:
            position = "NEUTRAL"
        signals.append({
            "ticker": ticker,
            "date": date,
            "alpha_score": round(alpha, 4),
            "confidence": round(confidence, 4),
            "position": position,
            "magnitude": round(magnitude if position != "NEUTRAL" else 0.0, 4),
        })

    return pd.DataFrame(signals).sort_values("alpha_score", ascending=False)


def print_signals_table(signals_df: pd.DataFrame, date: str) -> None:
    """Render a rich table of signals."""
    table = Table(
        title=f"🧠 NeuralAlpha Signals  —  {date}",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Ticker", style="bold white", justify="center", width=8)
    table.add_column("Position", justify="center", width=10)
    table.add_column("Alpha Score", justify="right", width=12)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Size", justify="right", width=8)
    table.add_column("Conviction", justify="left", width=20)

    for _, row in signals_df.iterrows():
        pos = row["position"]
        alpha = row["alpha_score"]
        conf = row["confidence"]
        mag = row["magnitude"]

        # Color by position
        if pos == "LONG":
            pos_str = "[bold green]● LONG[/bold green]"
            alpha_str = f"[green]{alpha:+.4f}[/green]"
        elif pos == "SHORT":
            pos_str = "[bold red]● SHORT[/bold red]"
            alpha_str = f"[red]{alpha:+.4f}[/red]"
        else:
            pos_str = "[grey50]○ NEUTRAL[/grey50]"
            alpha_str = f"[grey50]{alpha:+.4f}[/grey50]"

        # Conviction bar
        bar_len = int(conf * 15)
        conviction_bar = "[cyan]" + "█" * bar_len + "░" * (15 - bar_len) + "[/cyan]"

        table.add_row(
            row["ticker"],
            pos_str,
            alpha_str,
            f"{conf:.2%}",
            f"{mag:.0%}" if mag > 0 else "—",
            conviction_bar,
        )

    console.print(table)


def main():
    args = parse_args()
    from datetime import date as dt_date
    signal_date = args.date or dt_date.today().strftime("%Y-%m-%d")

    console.rule("[bold blue]NeuralAlpha — Neuro-Symbolic Investment Intelligence[/bold blue]")
    console.print(f"[dim]Tickers: {', '.join(args.tickers)}[/dim]")
    console.print(f"[dim]Date:    {signal_date}[/dim]\n")

    if args.pretrained:
        from neural_alpha import NeuralAlphaPipeline
        console.print(f"[green]Loading pretrained model from {args.pretrained}...[/green]")
        model = NeuralAlphaPipeline.from_pretrained(args.pretrained, device=args.device)
        with console.status("[bold green]Generating signals..."):
            signals_df = model.generate_signals(args.tickers, signal_date)
    else:
        signals_df = demo_without_pretrained(args.tickers, signal_date)

    if signals_df.empty:
        console.print("[red]No signals generated.[/red]")
        return

    print_signals_table(signals_df, signal_date)

    # Summary stats
    n_long = (signals_df["position"] == "LONG").sum()
    n_short = (signals_df["position"] == "SHORT").sum()
    n_neutral = (signals_df["position"] == "NEUTRAL").sum()

    console.print(
        f"\n[bold]Summary:[/bold]  "
        f"[green]{n_long} LONG[/green]  |  "
        f"[red]{n_short} SHORT[/red]  |  "
        f"[grey50]{n_neutral} NEUTRAL[/grey50]"
    )
    console.print(
        f"[dim]Mean confidence: {signals_df['confidence'].mean():.2%}  |  "
        f"Avg |alpha|: {signals_df['alpha_score'].abs().mean():.4f}[/dim]\n"
    )


if __name__ == "__main__":
    main()
