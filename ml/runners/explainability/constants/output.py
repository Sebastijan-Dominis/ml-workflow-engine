from dataclasses import dataclass

from ml.runners.explainability.constants.explainability_metrics_class import \
    ExplainabilityMetrics


@dataclass
class ExplainabilityOutput:
    explainability_metrics: ExplainabilityMetrics
    feature_lineage: list[dict]