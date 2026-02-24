"""
NeuralAlpha Pipeline
====================

End-to-end inference pipeline that wraps:
    MarketEncoder → CausalGraphEmbedder → AlphaTransformer → SignalSynthesizer

Designed to be loaded from pretrained checkpoints and used for
live or historical signal generation.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

from neural_alpha.encoder import MarketEncoder, MarketFeatureEngineer
from neural_alpha.causal import CausalEngine, CausalGraphEmbedder
from neural_alpha.transformer import AlphaTransformer
from neural_alpha.synthesizer import SignalSynthesizer
from neural_alpha.synthesizer.signal_head import AlphaSignal
from neural_alpha.utils.data_loader import MarketDataLoader

logger = logging.getLogger(__name__)


class NeuralAlphaPipeline(nn.Module):
    """
    Full NeuralAlpha inference pipeline.

    Args:
        n_features:   Number of market features (from feature engineer).
        latent_dim:   Encoder output dimension.
        causal_nodes: Number of causal graph nodes.
        causal_dim:   Causal embedding dimension.
        d_model:      Transformer model dimension.
        seq_len:      Lookback window (trading days).
        device:       'cpu' or 'cuda'.
    """

    def __init__(
        self,
        n_features: int = 32,
        latent_dim: int = 256,
        causal_nodes: int = 12,
        causal_dim: int = 64,
        d_model: int = 256,
        seq_len: int = 60,
        device: str = "cpu",
    ):
        super().__init__()

        self.seq_len = seq_len
        self.device = torch.device(device)

        # Sub-modules
        self.feature_engineer = MarketFeatureEngineer()
        self.encoder = MarketEncoder(n_features=n_features, latent_dim=latent_dim)
        self.causal_embedder = CausalGraphEmbedder(
            n_nodes=causal_nodes, embed_dim=causal_dim
        )
        self.transformer = AlphaTransformer(
            market_dim=latent_dim,
            causal_dim=causal_dim,
            d_model=d_model,
            seq_len=seq_len,
        )
        self.synthesizer = SignalSynthesizer(d_model=d_model)

        # Causal adjacency (loaded or learned)
        self.register_buffer(
            "causal_adj",
            torch.zeros(causal_nodes, causal_nodes),
        )

        self.to(self.device)

    def forward(
        self,
        features: torch.Tensor,
        causal_adj: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Full forward pass.

        Args:
            features:   (B, T, n_features)
            causal_adj: Optional (causal_nodes, causal_nodes) adjacency matrix.
                        If None, uses stored self.causal_adj.

        Returns:
            (alpha, confidence, magnitude): each (B,)
        """
        adj = causal_adj if causal_adj is not None else self.causal_adj

        # 1. Market encoding
        h = self.encoder(features)  # (B, T, latent_dim)

        # 2. Causal graph embedding
        causal_embed = self.causal_embedder(adj)  # (causal_dim,)

        # 3. Transformer
        z = self.transformer(h, causal_embed)  # (B, T, d_model)

        # 4. Signal synthesis
        alpha, confidence, magnitude = self.synthesizer(z)

        return alpha, confidence, magnitude

    def generate_signals(
        self,
        tickers: List[str],
        date: str,
        data_dir: str = "data/raw/",
        lookback_days: int = 120,
    ) -> pd.DataFrame:
        """
        Generate alpha signals for a list of tickers on a given date.

        Args:
            tickers:       List of ticker symbols.
            date:          Signal date "YYYY-MM-DD".
            data_dir:      Directory for cached market data.
            lookback_days: Days of history to load for feature computation.

        Returns:
            signals_df: DataFrame with columns [ticker, alpha_score, confidence,
                        position, magnitude, date].
        """
        self.eval()

        # Load market data
        end_dt = pd.Timestamp(date) + pd.Timedelta(days=5)  # small buffer
        start_dt = pd.Timestamp(date) - pd.Timedelta(days=lookback_days + 100)
        loader = MarketDataLoader(cache_dir=data_dir)
        market_data = loader.load(
            tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )

        signals = []
        with torch.no_grad():
            for ticker in tickers:
                if ticker not in market_data:
                    logger.warning(f"No data for {ticker}, skipping.")
                    continue

                df = market_data[ticker]

                # Feature engineering
                try:
                    features = self.feature_engineer.fit_transform(df)
                except Exception as e:
                    logger.warning(f"Feature engineering failed for {ticker}: {e}")
                    continue

                # Take last seq_len rows
                if len(features) < self.seq_len:
                    logger.warning(f"Insufficient data for {ticker}: {len(features)} < {self.seq_len}")
                    continue

                feat_window = features[-self.seq_len:]  # (T, n_features)
                feat_t = torch.tensor(feat_window, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
                feat_t = feat_t.to(self.device)

                # Run pipeline
                alpha, confidence, magnitude = self(feat_t)
                signal = self.synthesizer.decode_signal(
                    alpha=alpha.item(),
                    confidence=confidence.item(),
                    magnitude=magnitude.item(),
                    ticker=ticker,
                    date=date,
                )
                signals.append(signal)

        if not signals:
            logger.warning("No signals generated.")
            return pd.DataFrame()

        df_out = SignalSynthesizer.signals_to_dataframe(signals)
        df_out = df_out.sort_values("alpha_score", ascending=False).reset_index(drop=True)
        return df_out

    def save(self, checkpoint_dir: str) -> None:
        """Save model weights and config to checkpoint directory."""
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path / "model.pt")

        import json
        config = {
            "n_features": self.encoder.n_features,
            "latent_dim": self.encoder.latent_dim,
            "d_model": self.transformer.d_model,
            "seq_len": self.seq_len,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Pipeline saved to {checkpoint_dir}")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        device: str = "cpu",
    ) -> "NeuralAlphaPipeline":
        """Load a pretrained pipeline from a checkpoint directory."""
        import json
        path = Path(checkpoint_dir)

        with open(path / "config.json") as f:
            config = json.load(f)

        pipeline = cls(**config, device=device)
        state_dict = torch.load(path / "model.pt", map_location=device)
        pipeline.load_state_dict(state_dict)
        pipeline.eval()

        logger.info(f"Pipeline loaded from {checkpoint_dir}")
        return pipeline
