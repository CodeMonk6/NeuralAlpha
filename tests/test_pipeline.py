"""Integration tests for the full NeuralAlpha pipeline."""

import pytest
import torch
import numpy as np
from neural_alpha import NeuralAlphaPipeline
from neural_alpha.synthesizer.signal_head import AlphaSignal, SignalSynthesizer


class TestNeuralAlphaPipeline:
    @pytest.fixture
    def pipeline(self):
        return NeuralAlphaPipeline(
            n_features=32,
            latent_dim=128,
            causal_nodes=6,
            causal_dim=32,
            d_model=128,
            seq_len=30,
            device="cpu",
        )

    def test_forward_shape(self, pipeline):
        B, T, F = 4, 30, 32
        features = torch.randn(B, T, F)
        alpha, confidence, magnitude = pipeline(features)
        assert alpha.shape == (B,)
        assert confidence.shape == (B,)
        assert magnitude.shape == (B,)

    def test_alpha_range(self, pipeline):
        features = torch.randn(4, 30, 32)
        alpha, _, _ = pipeline(features)
        assert (alpha >= -1).all() and (alpha <= 1).all()

    def test_confidence_range(self, pipeline):
        features = torch.randn(4, 30, 32)
        _, confidence, _ = pipeline(features)
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_magnitude_range(self, pipeline):
        features = torch.randn(4, 30, 32)
        _, _, magnitude = pipeline(features)
        assert (magnitude >= 0).all() and (magnitude <= 1).all()

    def test_no_nan_output(self, pipeline):
        features = torch.randn(4, 30, 32)
        alpha, confidence, magnitude = pipeline(features)
        assert not torch.isnan(alpha).any()
        assert not torch.isnan(confidence).any()
        assert not torch.isnan(magnitude).any()

    def test_decode_signal(self, pipeline):
        synth = pipeline.synthesizer
        signal = synth.decode_signal(
            alpha=0.5,
            confidence=0.8,
            magnitude=0.6,
            ticker="AAPL",
            date="2024-01-15",
        )
        assert isinstance(signal, AlphaSignal)
        assert signal.position == "LONG"
        assert signal.ticker == "AAPL"

    def test_decode_neutral_low_confidence(self, pipeline):
        synth = pipeline.synthesizer
        signal = synth.decode_signal(
            alpha=0.5,
            confidence=0.3,  # below threshold
            magnitude=0.6,
            ticker="MSFT",
            date="2024-01-15",
        )
        assert signal.position == "NEUTRAL"
        assert signal.magnitude == 0.0

    def test_signals_to_dataframe(self):
        signals = [
            AlphaSignal("AAPL", "2024-01-01", 0.5, 0.8, "LONG", 0.6, 0.5),
            AlphaSignal("TSLA", "2024-01-01", -0.4, 0.75, "SHORT", 0.5, -0.4),
        ]
        df = SignalSynthesizer.signals_to_dataframe(signals)
        assert len(df) == 2
        assert "alpha_score" in df.columns
        assert "position" in df.columns

    def test_eval_mode_no_dropout_change(self, pipeline):
        """Repeated inference in eval mode should be deterministic."""
        pipeline.eval()
        features = torch.randn(2, 30, 32)
        with torch.no_grad():
            a1, c1, m1 = pipeline(features)
            a2, c2, m2 = pipeline(features)
        assert torch.allclose(a1, a2)
        assert torch.allclose(c1, c2)

    def test_save_load(self, pipeline, tmp_path):
        """Test checkpoint save/load round-trip."""
        checkpoint_dir = str(tmp_path / "test_ckpt")
        pipeline.save(checkpoint_dir)

        loaded = NeuralAlphaPipeline.from_pretrained(checkpoint_dir)
        loaded.eval()
        pipeline.eval()

        features = torch.randn(2, 30, 32)
        with torch.no_grad():
            a1, _, _ = pipeline(features)
            a2, _, _ = loaded(features)

        assert torch.allclose(a1, a2, atol=1e-5)
