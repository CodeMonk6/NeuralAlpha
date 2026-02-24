"""Tests for the Market Encoder module."""

import pytest
import torch
import numpy as np
import pandas as pd
from neural_alpha.encoder import MarketEncoder, MarketFeatureEngineer
from neural_alpha.encoder.attention import TemporalAttention, CrossFrequencyAttention


class TestMarketEncoder:
    @pytest.fixture
    def encoder(self):
        return MarketEncoder(n_features=32, latent_dim=256, hidden_dim=64, tcn_depth=3)

    @pytest.fixture
    def dummy_input(self):
        B, T, F = 4, 60, 32
        return torch.randn(B, T, F)

    def test_output_shape(self, encoder, dummy_input):
        B, T, F = dummy_input.shape
        h = encoder(dummy_input)
        assert h.shape == (B, T, 256), f"Expected (B, T, 256), got {h.shape}"

    def test_output_dim_property(self, encoder):
        assert encoder.output_dim == 256

    def test_no_nan_in_output(self, encoder, dummy_input):
        h = encoder(dummy_input)
        assert not torch.isnan(h).any(), "NaN detected in encoder output"

    def test_different_batch_sizes(self, encoder):
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 60, 32)
            out = encoder(x)
            assert out.shape[0] == batch_size

    def test_different_seq_lengths(self, encoder):
        for seq_len in [20, 40, 60, 120]:
            x = torch.randn(2, seq_len, 32)
            out = encoder(x)
            assert out.shape == (2, seq_len, 256)

    def test_gradient_flows(self, encoder, dummy_input):
        dummy_input.requires_grad_(True)
        h = encoder(dummy_input)
        loss = h.mean()
        loss.backward()
        assert dummy_input.grad is not None


class TestTemporalAttention:
    def test_self_attention_shape(self):
        attn = TemporalAttention(embed_dim=64, n_heads=4)
        x = torch.randn(2, 30, 64)
        out = attn(x)
        assert out.shape == (2, 30, 64)

    def test_causal_masking(self):
        """Output at position t should not depend on positions > t."""
        attn = TemporalAttention(embed_dim=64, n_heads=4)
        attn.eval()
        x = torch.randn(1, 10, 64)
        x2 = x.clone()
        x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :])  # perturb future

        with torch.no_grad():
            out1 = attn(x)
            out2 = attn(x2)

        # First 5 positions should be identical
        assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5)


class TestMarketFeatureEngineer:
    @pytest.fixture
    def sample_ohlcv(self):
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.randn(300) * 0.01)
        df = pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(300) * 0.005),
                "high": prices * (1 + np.abs(np.random.randn(300)) * 0.01),
                "low": prices * (1 - np.abs(np.random.randn(300)) * 0.01),
                "close": prices,
                "volume": np.random.randint(1_000_000, 10_000_000, 300),
            },
            index=dates,
        )
        return df

    def test_output_shape(self, sample_ohlcv):
        eng = MarketFeatureEngineer()
        features = eng.fit_transform(sample_ohlcv)
        assert features.ndim == 2
        assert features.shape[1] == 32  # n_features

    def test_no_nan(self, sample_ohlcv):
        eng = MarketFeatureEngineer()
        features = eng.fit_transform(sample_ohlcv)
        assert not np.isnan(features).any(), "NaN found in features"

    def test_feature_count(self, sample_ohlcv):
        eng = MarketFeatureEngineer()
        features = eng.fit_transform(sample_ohlcv)
        assert features.shape[1] == len(MarketFeatureEngineer.FEATURE_NAMES)

    def test_dtype(self, sample_ohlcv):
        eng = MarketFeatureEngineer()
        features = eng.fit_transform(sample_ohlcv)
        assert features.dtype == np.float32
