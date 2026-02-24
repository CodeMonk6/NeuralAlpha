"""
Market Encoder
==============

Multi-resolution temporal encoder for OHLCV + alternative data.
Produces a dense latent market state h_t ∈ ℝ^{latent_dim} per asset per timestep.

Architecture:
    Input features (n_features) 
        → Multi-resolution TCN branches (daily / weekly / monthly horizons)
        → Cross-frequency attention fusion
        → Linear projection to latent_dim
        → LayerNorm output

Paper inspiration:
    - Temporal Convolutional Networks (Bai et al., 2018)
    - Informer: Beyond Efficient Transformer (Zhou et al., 2021)
"""

import torch
import torch.nn as nn
from typing import Optional, List
from neural_alpha.encoder.attention import CrossFrequencyAttention


class TemporalBlock(nn.Module):
    """
    One causal dilated conv block: Conv1D → GroupNorm → GELU → Dropout.
    
    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Kernel size for the convolution.
        dilation:     Dilation factor (controls receptive field).
        dropout:      Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        out = out[:, :, : x.size(2)]  # trim causal padding
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return out + self.residual(x)


class TCNBranch(nn.Module):
    """
    Multi-level dilated TCN for a single temporal resolution.
    
    Receptive field grows exponentially with depth via doubling dilation.
    e.g., kernel=3, depth=4 → RF = 1 + (3-1)*(1+2+4+8) = 31 timesteps.
    
    Args:
        in_channels:  Input feature dimension.
        hidden_dim:   Hidden channel width.
        depth:        Number of TemporalBlocks (each doubles dilation).
        dropout:      Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for i in range(depth):
            c_in = in_channels if i == 0 else hidden_dim
            layers.append(
                TemporalBlock(
                    c_in,
                    hidden_dim,
                    kernel_size=3,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MarketEncoder(nn.Module):
    """
    Full Market State Encoder.

    Encodes a sequence of market features for one asset across multiple
    temporal resolutions (daily, weekly, monthly) into a unified latent
    representation h_t of dimension `latent_dim`.

    Args:
        n_features:   Number of input features per timestep.
        latent_dim:   Output latent dimension (default: 256).
        hidden_dim:   Hidden channel width for TCN branches.
        tcn_depth:    Depth (# blocks) per TCN branch.
        n_resolutions: Number of temporal resolutions (default: 3).
        dropout:      Dropout rate.

    Shapes:
        Input  x: (batch, seq_len, n_features)
        Output h: (batch, seq_len, latent_dim)
    """

    def __init__(
        self,
        n_features: int = 32,
        latent_dim: int = 256,
        hidden_dim: int = 64,
        tcn_depth: int = 4,
        n_resolutions: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_resolutions = n_resolutions

        # Downsampling strides for each resolution: 1x (daily), 5x (weekly), 21x (monthly)
        self.strides = [1, 5, 21][:n_resolutions]

        # One TCN branch per resolution
        self.tcn_branches = nn.ModuleList(
            [
                TCNBranch(n_features, hidden_dim, tcn_depth, dropout)
                for _ in range(n_resolutions)
            ]
        )

        # Cross-resolution fusion
        self.fusion = CrossFrequencyAttention(
            embed_dim=hidden_dim,
            n_heads=4,
            n_resolutions=n_resolutions,
        )

        # Project fused representation to latent_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * n_resolutions, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def _downsample(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        """Pool x along time dimension by `stride` using average pooling."""
        if stride == 1:
            return x
        # x: (B, C, T) — apply adaptive avg pool then interpolate back to T
        T = x.size(2)
        pooled = nn.functional.avg_pool1d(x, kernel_size=stride, stride=1, padding=stride // 2)
        return pooled[:, :, :T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, n_features)

        Returns:
            h: Tensor of shape (batch, seq_len, latent_dim)
        """
        B, T, F = x.shape
        x_t = x.permute(0, 2, 1)  # (B, F, T) for Conv1d

        branch_outputs = []
        for branch, stride in zip(self.tcn_branches, self.strides):
            x_s = self._downsample(x_t, stride)   # multi-resolution signal
            h_s = branch(x_s)                      # (B, hidden_dim, T)
            branch_outputs.append(h_s.permute(0, 2, 1))  # (B, T, hidden_dim)

        # Cross-frequency fusion
        fused = self.fusion(branch_outputs)  # (B, T, hidden_dim * n_resolutions)

        # Project to latent space
        h = self.proj(fused)  # (B, T, latent_dim)
        return self.dropout(h)

    @property
    def output_dim(self) -> int:
        return self.latent_dim
