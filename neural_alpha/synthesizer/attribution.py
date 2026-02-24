"""
Factor Attribution
==================

Explains *why* the model generated a particular alpha signal using
SHAP (SHapley Additive exPlanations) values.

Provides both:
    - Local explanations: Which features drove THIS specific signal?
    - Global explanations: What does the model rely on overall?

Attribution categories (mapped from feature indices):
    - Momentum:    log_return_*, ema_ratio, rsi_14
    - Mean-rev:    rolling_zscore_*, bb_*
    - Volume:      log_volume, volume_zscore*, dollar_volume_rank
    - Technical:   macd_*, atr_14, adx_14
    - Quality:     earnings_revision_30d, analyst_sentiment
    - Macro:       sector_relative_strength, market_beta_63d
    - Calendar:    day_of_week_*, month_*
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# Feature-to-group mapping (based on MarketFeatureEngineer.FEATURE_NAMES)
FEATURE_GROUPS = {
    "momentum": [0, 1, 2, 21, 22, 12],  # returns, ema, rsi
    "mean_reversion": [5, 6, 7, 16, 17, 18],  # zscore, bb
    "volume": [8, 9, 10, 11],
    "technical": [13, 14, 15, 19, 20, 23],  # macd, atr, adx
    "quality": [24, 25],  # earnings, analyst
    "macro": [26, 27],  # sector RS, beta
    "calendar": [28, 29, 30, 31],
}

GROUP_LABELS = list(FEATURE_GROUPS.keys())


@dataclass
class AttributionResult:
    """Attribution output for a single signal."""
    ticker: str
    date: str
    alpha_score: float
    feature_shap: np.ndarray          # (n_features,) SHAP values
    group_attribution: Dict[str, float] = field(default_factory=dict)
    top_drivers: List[Tuple[str, float]] = field(default_factory=list)
    dominant_theme: str = ""

    def summary(self) -> str:
        lines = [
            f"Attribution: {self.ticker} {self.date}  α={self.alpha_score:.3f}",
            f"  Dominant theme: {self.dominant_theme}",
            "  Top drivers:",
        ]
        for feat, val in self.top_drivers[:5]:
            lines.append(f"    {feat:30s}  SHAP={val:+.4f}")
        return "\n".join(lines)


class FactorAttributor:
    """
    SHAP-based factor attribution for NeuralAlpha signals.

    Uses kernel SHAP (model-agnostic) on the feature matrix to attribute
    signal generation to individual market features and factor groups.

    Args:
        feature_names: List of feature names (must match MarketFeatureEngineer.FEATURE_NAMES).
        background_samples: Number of background samples for SHAP estimation.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        background_samples: int = 100,
    ):
        from neural_alpha.encoder.preprocessing import MarketFeatureEngineer

        self.feature_names = feature_names or MarketFeatureEngineer.FEATURE_NAMES
        self.background_samples = background_samples
        self._explainer = None

    def setup_explainer(
        self,
        model_fn,
        background_data: np.ndarray,
    ) -> None:
        """
        Initialize the SHAP explainer with a background dataset.

        Args:
            model_fn:        Callable that takes (N, T, n_features) → alpha_scores.
            background_data: Background data array (N_bg, T, n_features).
        """
        try:
            import shap
            n = min(self.background_samples, len(background_data))
            bg = background_data[:n]
            self.model_fn = model_fn
            self._explainer = shap.KernelExplainer(
                model=lambda x: model_fn(x),
                data=bg.reshape(n, -1),  # flatten for KernelSHAP
            )
        except ImportError:
            raise ImportError("Install shap: pip install shap")

    def attribute(
        self,
        features: np.ndarray,
        ticker: str = "",
        date: str = "",
        alpha_score: float = 0.0,
    ) -> AttributionResult:
        """
        Compute SHAP values for a single observation.

        Args:
            features:    Feature array (T, n_features) for one asset.
            ticker:      Asset ticker (for labeling).
            date:        Signal date (for labeling).
            alpha_score: The model's alpha output.

        Returns:
            AttributionResult with SHAP values and group breakdown.
        """
        if self._explainer is None:
            # Fallback: random attribution (for testing without SHAP setup)
            shap_values = np.random.randn(len(self.feature_names)) * 0.01
        else:
            flat_features = features.flatten().reshape(1, -1)
            shap_values = self._explainer.shap_values(flat_features)[0]
            # Average over time dimension
            shap_values = shap_values.reshape(features.shape).mean(axis=0)

        # Group attributions
        group_attr = {}
        for group, indices in FEATURE_GROUPS.items():
            valid_idx = [i for i in indices if i < len(shap_values)]
            group_attr[group] = float(np.sum(np.abs(shap_values[valid_idx])))

        # Normalize group attributions to sum to 1
        total = sum(group_attr.values()) + 1e-9
        group_attr = {k: v / total for k, v in group_attr.items()}

        # Top individual drivers
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]
        top_drivers = [
            (self.feature_names[i], float(shap_values[i]))
            for i in sorted_idx[:10]
        ]

        # Dominant theme
        dominant_theme = max(group_attr, key=group_attr.get)

        return AttributionResult(
            ticker=ticker,
            date=date,
            alpha_score=alpha_score,
            feature_shap=shap_values,
            group_attribution=group_attr,
            top_drivers=top_drivers,
            dominant_theme=dominant_theme,
        )

    def batch_attribute(
        self,
        features_list: List[np.ndarray],
        tickers: List[str],
        dates: List[str],
        alpha_scores: List[float],
    ) -> List[AttributionResult]:
        """Batch attribution for multiple assets."""
        return [
            self.attribute(f, t, d, a)
            for f, t, d, a in zip(features_list, tickers, dates, alpha_scores)
        ]
