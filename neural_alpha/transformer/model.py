"""
AlphaTransformer
================

6-layer, 8-head Transformer that attends over:
    - Market state sequence h_t ∈ ℝ^{market_dim} (from MarketEncoder)
    - Causal graph embedding c ∈ ℝ^{causal_dim} (from CausalGraphEmbedder)

The causal embedding is injected at each layer via cross-attention,
allowing the model to continuously condition on structural market dynamics.

Output: per-timestep latent z_t ∈ ℝ^{d_model} → fed to SignalSynthesizer.
"""

import torch
import torch.nn as nn
from neural_alpha.transformer.layers import AlphaAttentionLayer, FeedForward
from neural_alpha.transformer.positional import TemporalPositionalEncoding


class AlphaTransformer(nn.Module):
    """
    Causal-conditioned Transformer for alpha signal generation.

    Args:
        market_dim:   Dimension of market encoder output (h_t).
        causal_dim:   Dimension of causal graph embedding.
        d_model:      Internal model dimension.
        n_heads:      Number of self-attention heads.
        n_layers:     Number of Transformer layers.
        d_ff:         Feed-forward hidden dimension.
        max_seq_len:  Maximum sequence length (trading days in context window).
        dropout:      Dropout probability.
    """

    def __init__(
        self,
        market_dim: int = 256,
        causal_dim: int = 64,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Project market state to model dimension
        self.market_proj = nn.Linear(market_dim, d_model)

        # Project causal embedding to model dimension (for cross-attention key/value)
        self.causal_proj = nn.Linear(causal_dim, d_model)

        # Temporal positional encoding
        self.pos_enc = TemporalPositionalEncoding(d_model, max_seq_len, dropout)

        # Stack of Transformer layers, each with self-attn + causal cross-attn + FFN
        self.layers = nn.ModuleList(
            [
                AlphaTransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        market_state: torch.Tensor,
        causal_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            market_state: (B, T, market_dim) — output of MarketEncoder
            causal_embed: (B, causal_dim) or (causal_dim,) — output of CausalGraphEmbedder

        Returns:
            z: (B, T, d_model) — latent representation for signal synthesis
        """
        # Ensure causal_embed has batch dimension
        if causal_embed.dim() == 1:
            causal_embed = causal_embed.unsqueeze(0).expand(market_state.size(0), -1)

        # Project inputs
        x = self.market_proj(market_state)   # (B, T, d_model)
        c = self.causal_proj(causal_embed)   # (B, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Expand causal embedding to (B, 1, d_model) for cross-attention
        c = c.unsqueeze(1)

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, c)

        return self.norm(x)

    @property
    def output_dim(self) -> int:
        return self.d_model


class AlphaTransformerLayer(nn.Module):
    """
    Single AlphaTransformer layer.

    Sub-layers:
        1. Causal self-attention (over time)
        2. Cross-attention conditioning on causal graph embedding
        3. Feed-forward network
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = AlphaAttentionLayer(d_model, n_heads, dropout, causal=True)
        self.cross_attn = AlphaAttentionLayer(d_model, n_heads, dropout, causal=False)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, T, d_model)
            causal_context: (B, 1, d_model) — causal graph conditioning

        Returns:
            out: (B, T, d_model)
        """
        # 1. Causal self-attention
        x = self.norm1(x + self.self_attn(x, x, x))

        # 2. Cross-attention: query=x, key/value=causal_context
        x = self.norm2(x + self.cross_attn(x, causal_context, causal_context))

        # 3. Feed-forward
        x = self.norm3(x + self.ffn(x))
        return x
