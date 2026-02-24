"""
Custom Attention & Feed-Forward Layers
=======================================

Modular building blocks for the AlphaTransformer.

- AlphaAttentionLayer: Scaled dot-product multi-head attention,
                       supports both causal (masked) and full attention.
- FeedForward:         Standard FFN with GELU activation and dropout.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AlphaAttentionLayer(nn.Module):
    """
    Multi-head scaled dot-product attention.

    Supports both:
        - Self-attention (query=key=value=x) with causal masking
        - Cross-attention (query from x, key/value from context)

    Args:
        d_model:  Model embedding dimension.
        n_heads:  Number of attention heads.
        dropout:  Attention weight dropout.
        causal:   If True, apply causal (lower triangular) mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.causal = causal

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, d_model)
            key:   (B, T_k, d_model)
            value: (B, T_k, d_model)
            mask:  Optional (T_q, T_k) boolean mask (True = ignore)

        Returns:
            out: (B, T_q, d_model)
        """
        B, T_q, _ = query.shape
        T_k = key.size(1)

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            B, T, D = x.shape
            return x.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        Q = split_heads(self.W_q(query))  # (B, H, T_q, d)
        K = split_heads(self.W_k(key))    # (B, H, T_k, d)
        V = split_heads(self.W_v(value))  # (B, H, T_k, d)

        attn_weights = (Q @ K.transpose(-2, -1)) / self.scale  # (B, H, T_q, T_k)

        # Apply causal mask
        if self.causal and T_q == T_k:
            causal_mask = torch.triu(
                torch.ones(T_q, T_k, device=query.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Apply external mask
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = attn_weights @ V  # (B, H, T_q, d)
        out = out.permute(0, 2, 1, 3).reshape(B, T_q, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Architecture: Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)

    Args:
        d_model:  Input/output dimension.
        d_ff:     Hidden dimension (typically 4 * d_model).
        dropout:  Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
