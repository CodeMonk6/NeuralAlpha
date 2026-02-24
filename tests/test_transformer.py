"""Tests for the AlphaTransformer module."""

import pytest
import torch
from neural_alpha.transformer import AlphaTransformer
from neural_alpha.transformer.layers import AlphaAttentionLayer, FeedForward
from neural_alpha.transformer.positional import TemporalPositionalEncoding


class TestAlphaTransformer:
    @pytest.fixture
    def transformer(self):
        return AlphaTransformer(
            market_dim=256,
            causal_dim=64,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=60,
        )

    def test_output_shape(self, transformer):
        B, T = 4, 60
        market_state = torch.randn(B, T, 256)
        causal_embed = torch.randn(64)
        z = transformer(market_state, causal_embed)
        assert z.shape == (B, T, 128)

    def test_batched_causal_embed(self, transformer):
        B, T = 4, 30
        market_state = torch.randn(B, T, 256)
        causal_embed = torch.randn(B, 64)  # batched causal embed
        z = transformer(market_state, causal_embed)
        assert z.shape == (B, T, 128)

    def test_no_nan(self, transformer):
        market_state = torch.randn(2, 60, 256)
        causal_embed = torch.randn(64)
        z = transformer(market_state, causal_embed)
        assert not torch.isnan(z).any()

    def test_output_dim_property(self, transformer):
        assert transformer.output_dim == 128

    def test_gradient_flow(self, transformer):
        market_state = torch.randn(2, 30, 256, requires_grad=True)
        causal_embed = torch.randn(64)
        z = transformer(market_state, causal_embed)
        z.mean().backward()
        assert market_state.grad is not None


class TestAlphaAttentionLayer:
    def test_self_attention_shape(self):
        attn = AlphaAttentionLayer(d_model=64, n_heads=4)
        x = torch.randn(2, 30, 64)
        out = attn(x, x, x)
        assert out.shape == (2, 30, 64)

    def test_cross_attention_shape(self):
        attn = AlphaAttentionLayer(d_model=64, n_heads=4, causal=False)
        query = torch.randn(2, 30, 64)
        context = torch.randn(2, 1, 64)
        out = attn(query, context, context)
        assert out.shape == (2, 30, 64)

    def test_causal_masking(self):
        attn = AlphaAttentionLayer(d_model=32, n_heads=2, causal=True)
        attn.eval()
        x = torch.randn(1, 20, 32)
        x2 = x.clone()
        x2[:, 10:, :] = torch.randn(1, 10, 32)  # perturb future

        with torch.no_grad():
            out1 = attn(x, x, x)
            out2 = attn(x2, x2, x2)

        # Past outputs should be unchanged by future perturbation
        assert torch.allclose(out1[:, :10, :], out2[:, :10, :], atol=1e-4)


class TestTemporalPositionalEncoding:
    def test_output_shape(self):
        pe = TemporalPositionalEncoding(d_model=128, max_len=60)
        x = torch.randn(4, 30, 128)
        out = pe(x)
        assert out.shape == (4, 30, 128)

    def test_sequence_shorter_than_max(self):
        pe = TemporalPositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(2, 20, 64)
        out = pe(x)
        assert out.shape == (2, 20, 64)

    def test_no_nan(self):
        pe = TemporalPositionalEncoding(d_model=64, max_len=60)
        x = torch.randn(2, 60, 64)
        out = pe(x)
        assert not torch.isnan(out).any()
