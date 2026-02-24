"""
Temporal & Cross-Frequency Attention
=====================================

Custom attention modules for the Market Encoder.

- TemporalAttention:       Standard causal multi-head self-attention over time.
- CrossFrequencyAttention: Fuses representations from different temporal resolutions
                           via cross-attention, producing a unified representation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TemporalAttention(nn.Module):
    """
    Causal multi-head self-attention over a temporal sequence.

    Uses an additive causal mask to prevent the model from attending to
    future timesteps (autoregressive constraint).

    Args:
        embed_dim:  Embedding dimensionality (d_model).
        n_heads:    Number of attention heads.
        dropout:    Attention dropout probability.
    """

    def __init__(self, embed_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D)
            mask: optional (T, T) boolean mask (True = mask out)

        Returns:
            out: (B, T, D)
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        QKV = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        Q, K, V = QKV.unbind(dim=2)
        Q = Q.permute(0, 2, 1, 3)  # (B, H, T, d)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        attn = (Q @ K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Causal mask (default if none provided)
        if mask is None:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
        else:
            causal_mask = mask

        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.out_proj(out)
        return self.resid_drop(out) + residual


class CrossFrequencyAttention(nn.Module):
    """
    Fuses branch representations from multiple temporal resolutions.

    Each branch (daily, weekly, monthly) is treated as a separate "view"
    of the same asset. This module cross-attends between views so each
    resolution can attend to information in the others.

    Args:
        embed_dim:      Embedding dim per branch.
        n_heads:        Number of cross-attention heads.
        n_resolutions:  Number of temporal resolution branches.
        dropout:        Dropout probability.

    Input:
        branches: List of n_resolutions tensors, each (B, T, embed_dim)

    Output:
        fused: (B, T, embed_dim * n_resolutions)  — concatenation of all updated views
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        n_resolutions: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_resolutions = n_resolutions

        # One cross-attention layer per pair of resolutions (query from i, key/value from j)
        # For simplicity: each resolution attends over ALL others (including itself)
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim,
                    n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_resolutions)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_resolutions)])

    def forward(self, branches: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            branches: List of tensors [(B, T, D), ...]

        Returns:
            fused: (B, T, D * n_resolutions)
        """
        # Stack all branches along a new dimension for key/value: (B, T, n_res, D)
        # Each branch queries the mean of all branches as context
        context = torch.stack(branches, dim=2)  # (B, T, n_res, D)
        B, T, n_res, D = context.shape
        context_flat = context.reshape(B, T * n_res, D)  # (B, T*n_res, D)

        updated = []
        for i, (attn_layer, norm) in enumerate(zip(self.cross_attn, self.norms)):
            q = branches[i]  # (B, T, D)
            out, _ = attn_layer(query=q, key=context_flat, value=context_flat)
            updated.append(norm(out + q))  # residual + norm

        return torch.cat(updated, dim=-1)  # (B, T, D * n_resolutions)
