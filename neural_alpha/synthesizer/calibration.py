"""
Confidence Calibration
=======================

Neural networks tend to be overconfident. This module applies
**temperature scaling** (Guo et al., 2017) to post-hoc calibrate
the confidence estimates from the SignalSynthesizer.

After calibration, the confidence score approximates the empirical
hit rate (accuracy) of the signal — i.e., a confidence of 0.7 should
correspond to ~70% of signals in that bucket being correct.

Usage:
    calibrator = ConfidenceCalibrator()
    calibrator.fit(raw_confidences, true_labels)
    calibrated = calibrator.transform(raw_confidences)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class TemperatureScaler(nn.Module):
    """
    Temperature scaling for confidence calibration.
    
    Learns a single scalar T such that:
        calibrated_confidence = sigmoid(logit(raw_conf) / T)

    T > 1  → softer (lower) confidences (fixes overconfidence)
    T < 1  → harder (higher) confidences (fixes underconfidence)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits / self.temperature)


class ConfidenceCalibrator:
    """
    Post-hoc confidence calibrator for alpha signals.

    Supports three methods:
        - 'temperature': Temperature scaling (parametric, fast).
        - 'isotonic':    Isotonic regression (non-parametric, flexible).
        - 'platt':       Platt scaling via logistic regression.

    Args:
        method: Calibration method ('temperature', 'isotonic', 'platt').
    """

    def __init__(self, method: str = "temperature"):
        assert method in ("temperature", "isotonic", "platt")
        self.method = method
        self._fitted = False

        if method == "temperature":
            self._model = TemperatureScaler()
        elif method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip")
        elif method == "platt":
            self._model = LogisticRegression(C=1.0)

    def fit(
        self,
        raw_confidences: np.ndarray,
        true_labels: np.ndarray,
    ) -> "ConfidenceCalibrator":
        """
        Fit the calibrator on a held-out validation set.

        Args:
            raw_confidences: (N,) array of raw confidence outputs ∈ (0, 1).
            true_labels:     (N,) binary array (1=signal correct, 0=signal wrong).

        Returns:
            self
        """
        if self.method == "temperature":
            self._fit_temperature(raw_confidences, true_labels)
        elif self.method == "isotonic":
            self._model.fit(raw_confidences, true_labels)
        elif self.method == "platt":
            self._model.fit(raw_confidences.reshape(-1, 1), true_labels)

        self._fitted = True
        return self

    def transform(self, raw_confidences: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw confidence scores.

        Args:
            raw_confidences: (N,) array ∈ (0, 1).

        Returns:
            calibrated: (N,) array ∈ (0, 1).
        """
        if not self._fitted:
            return raw_confidences  # passthrough if not fitted

        if self.method == "temperature":
            logits = np.log(raw_confidences.clip(1e-7, 1 - 1e-7) / (1 - raw_confidences.clip(1e-7, 1 - 1e-7)))
            logits_t = torch.tensor(logits, dtype=torch.float32)
            with torch.no_grad():
                return self._model(logits_t).numpy()
        elif self.method == "isotonic":
            return self._model.predict(raw_confidences)
        elif self.method == "platt":
            return self._model.predict_proba(raw_confidences.reshape(-1, 1))[:, 1]

    def _fit_temperature(
        self, raw_confidences: np.ndarray, true_labels: np.ndarray
    ) -> None:
        """Fit temperature via NLL minimization on validation set."""
        eps = 1e-7
        logits = np.log(raw_confidences.clip(eps, 1 - eps) / (1 - raw_confidences.clip(eps, 1 - eps)))

        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(true_labels, dtype=torch.float32)

        optimizer = torch.optim.LBFGS(
            [self._model.temperature], lr=0.01, max_iter=100
        )

        def _eval():
            optimizer.zero_grad()
            scaled_probs = self._model(logits_t)
            loss = nn.functional.binary_cross_entropy(scaled_probs, labels_t)
            loss.backward()
            return loss

        optimizer.step(_eval)

    def expected_calibration_error(
        self,
        confidences: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Lower is better. ECE = 0 means perfectly calibrated.

        Args:
            confidences: (N,) predicted confidence scores.
            labels:      (N,) true binary labels.
            n_bins:      Number of equal-width bins.

        Returns:
            ece: Scalar ECE value.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(confidences)

        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_conf = confidences[mask].mean()
            bin_acc = labels[mask].mean()
            ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

        return float(ece)
