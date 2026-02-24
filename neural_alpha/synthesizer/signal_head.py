"""
Signal Synthesizer
==================

Maps the AlphaTransformer's latent representation z_t to actionable
investment signals.

Outputs:
    alpha_score  ∈ (-1, 1)  — directional signal (positive = long, negative = short)
    confidence   ∈ (0, 1)   — calibrated confidence in the signal
    position     ∈ {LONG, SHORT, NEUTRAL} — discretized position recommendation
    magnitude    ∈ [0, 1]   — suggested position size (fraction of max weight)

Architecture:
    z_t (d_model)
        → Linear(d_model, d_model // 2) → GELU → Dropout
        → Linear(d_model // 2, d_model // 4) → GELU
        → Signal Head (linear → tanh)      → alpha_score
        → Confidence Head (linear → sigmoid) → raw_confidence
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class AlphaSignal:
    """Structured output of the SignalSynthesizer."""
    ticker: str
    date: str
    alpha_score: float       # (-1, 1)
    confidence: float        # (0, 1)
    position: str            # LONG / SHORT / NEUTRAL
    magnitude: float         # (0, 1) suggested position fraction
    raw_logit: float         # pre-activation logit (for debugging)

    def __repr__(self) -> str:
        return (
            f"AlphaSignal({self.ticker} | {self.date} | "
            f"{self.position} score={self.alpha_score:.3f} "
            f"conf={self.confidence:.3f} size={self.magnitude:.2f})"
        )

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "alpha_score": self.alpha_score,
            "confidence": self.confidence,
            "position": self.position,
            "magnitude": self.magnitude,
        }


class SignalSynthesizer(nn.Module):
    """
    Final signal output head of the NeuralAlpha pipeline.

    Takes the last token representation z_T from the Transformer
    and produces a structured alpha signal.

    Args:
        d_model:              Input latent dimension (from AlphaTransformer).
        long_threshold:       Alpha score above which → LONG.
        short_threshold:      Alpha score below which → SHORT.
        confidence_threshold: Minimum confidence to emit a non-NEUTRAL signal.
        dropout:              Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        long_threshold: float = 0.15,
        short_threshold: float = -0.15,
        confidence_threshold: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.confidence_threshold = confidence_threshold

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
        )

        # Alpha signal head: scalar in (-1, 1)
        self.alpha_head = nn.Linear(d_model // 4, 1)

        # Confidence head: scalar in (0, 1)
        self.conf_head = nn.Sequential(
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Magnitude head: position sizing (0 to 1)
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Args:
            z: Latent representation (B, T, d_model).
               Uses the LAST timestep z[:, -1, :] for signal generation.

        Returns:
            (alpha_score, confidence, magnitude): each (B,) tensors
        """
        z_last = z[:, -1, :]  # (B, d_model)
        h = self.trunk(z_last)  # (B, d_model // 4)

        alpha_raw = self.alpha_head(h).squeeze(-1)  # (B,) — unbounded
        alpha = torch.tanh(alpha_raw)               # (B,) ∈ (-1, 1)
        confidence = self.conf_head(h).squeeze(-1)  # (B,) ∈ (0, 1)
        magnitude = self.magnitude_head(h).squeeze(-1)  # (B,) ∈ (0, 1)

        return alpha, confidence, magnitude

    def decode_signal(
        self,
        alpha: float,
        confidence: float,
        magnitude: float,
        ticker: str = "",
        date: str = "",
    ) -> AlphaSignal:
        """
        Convert raw float outputs into a structured AlphaSignal.
        
        Applies thresholds and confidence gating.
        """
        if confidence < self.confidence_threshold:
            position = "NEUTRAL"
        elif alpha > self.long_threshold:
            position = "LONG"
        elif alpha < self.short_threshold:
            position = "SHORT"
        else:
            position = "NEUTRAL"

        return AlphaSignal(
            ticker=ticker,
            date=date,
            alpha_score=float(alpha),
            confidence=float(confidence),
            position=position,
            magnitude=float(magnitude) if position != "NEUTRAL" else 0.0,
            raw_logit=float(alpha),
        )

    @classmethod
    def signals_to_dataframe(cls, signals: list) -> pd.DataFrame:
        """Convert a list of AlphaSignal objects to a pandas DataFrame."""
        return pd.DataFrame([s.to_dict() for s in signals])
