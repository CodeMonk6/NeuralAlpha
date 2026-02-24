"""
NeuralAlpha — Neuro-Symbolic Investment Intelligence Platform
=============================================================

A research-grade platform combining Transformer-based market modeling
with causal inference to generate validated alpha signals.

Usage:
    from neural_alpha import NeuralAlphaPipeline

    pipeline = NeuralAlphaPipeline.from_pretrained("checkpoints/full/")
    signals = pipeline.generate_signals(["AAPL", "MSFT"], date="2024-01-01")
"""

from neural_alpha.pipeline import NeuralAlphaPipeline
from neural_alpha.encoder import MarketEncoder
from neural_alpha.causal import CausalEngine
from neural_alpha.transformer import AlphaTransformer
from neural_alpha.synthesizer import SignalSynthesizer

__version__ = "0.1.0"
__author__ = "Sourabh Sharma"

__all__ = [
    "NeuralAlphaPipeline",
    "MarketEncoder",
    "CausalEngine",
    "AlphaTransformer",
    "SignalSynthesizer",
]
