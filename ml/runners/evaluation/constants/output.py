"""Typed output model for evaluation runner results."""

from dataclasses import dataclass

from ml.modeling.models.feature_lineage import FeatureLineage
from ml.runners.evaluation.models.predictions import PredictionArtifacts


@dataclass
class EvaluateOutput:
    """Evaluation metrics, prediction artifacts, and feature lineage payload."""

    metrics: dict[str, dict[str, float]]
    prediction_dfs: PredictionArtifacts
    lineage: list[FeatureLineage]
