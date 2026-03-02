"""Typed output model returned by explainability runner implementations."""

from dataclasses import dataclass

from ml.runners.explainability.constants.explainability_metrics_class import \
    ExplainabilityMetrics


@dataclass
class ExplainabilityOutput:
    """Explainability metrics and feature lineage payload for persistence."""

    explainability_metrics: ExplainabilityMetrics
    feature_lineage: list[dict]