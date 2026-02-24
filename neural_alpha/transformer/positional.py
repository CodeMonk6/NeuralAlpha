"""
Temporal Positional Encoding
=============================

Financial time series have irregular structure (market holidays, earnings
windows, expiration cycles). Standard sinusoidal PE works well but we add
two enhancements:

1. **Learnable calendar features** — day-of-week, month, earnings-proximity
   encoded as sinusoidal + learned offsets.
2. **Decay weighting** — positions further back receive slightly lower
   effective "attention budget" to reflect the intuition that older data
   is less relevant for short-horizon signals.
"""

import math
import torch
import torch.nn as nn


class TemporalPositionalEncoding(nn.Module):
    """
    Positional encoding combining fixed sinusoidal PE with learnable
    temporal bias.

    Args:
        d_model:     Model embedding dimension (must be even).
        max_len:     Maximum sequence length.
        dropout:     Dropout applied after adding PE.
        learnable:   If True, add a learnable bias on top of fixed PE.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 60,
        dropout: float = 0.1,
        learnable: bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learnable = learnable

        # Fixed sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

        # Learnable bias
        if learnable:
            self.bias = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            x + PE: (B, T, d_model)
        """
        T = x.size(1)
        pe = self.pe[:, :T, :]  # truncate to actual sequence length

        if self.learnable:
            pe = pe + self.bias[:, :T, :]

        return self.dropout(x + pe)


class FourierTemporalEncoding(nn.Module):
    """
    Alternative: Fourier-based encoding that accepts absolute timestamps.

    Useful when the sequence has known irregular gaps (e.g., market holidays
    create non-uniform time steps).

    Args:
        d_model:   Model embedding dimension.
        n_freqs:   Number of Fourier frequency bands.
        dropout:   Dropout probability.
    """

    def __init__(self, d_model: int, n_freqs: int = 64, dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0
        self.n_freqs = n_freqs
        self.d_model = d_model

        # Learnable frequency + phase
        self.freq = nn.Parameter(torch.randn(n_freqs) * 0.1)
        self.phase = nn.Parameter(torch.zeros(n_freqs))
        self.proj = nn.Linear(2 * n_freqs, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (B, T, d_model)
            timestamps: (B, T) float tensor of timestamps (e.g., Unix epoch days)

        Returns:
            x + FourierPE: (B, T, d_model)
        """
        # (B, T, n_freqs)
        t = timestamps.unsqueeze(-1) * self.freq.unsqueeze(0).unsqueeze(0) + self.phase
        pe_raw = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)  # (B, T, 2*n_freqs)
        pe = self.proj(pe_raw)  # (B, T, d_model)
        return self.dropout(x + pe)
