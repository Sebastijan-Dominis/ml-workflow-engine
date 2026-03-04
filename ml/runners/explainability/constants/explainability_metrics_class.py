"""Typed explainability metric containers produced by explainers."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class ExplainabilityMetrics:
    """Top-k explainability tables for model-level interpretation outputs."""

    top_k_feature_importances: pd.DataFrame | None = None
    top_k_shap_importances: pd.DataFrame | None = None
