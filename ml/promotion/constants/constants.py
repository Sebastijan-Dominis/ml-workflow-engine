"""Shared constants and typed result containers for promotion workflows."""

import logging
from dataclasses import dataclass
from typing import Literal

from ml.metadata.schemas.runners.evaluation import EvaluationMetadata
from ml.metadata.schemas.runners.explainability import ExplainabilityMetadata
from ml.metadata.schemas.runners.training import TrainingMetadata
from ml.promotion.config.promotion_thresholds import Direction, MetricName, MetricSet

logger = logging.getLogger(__name__)

Stage = Literal["staging", "production"]

@dataclass
class RunnersMetadata:
    """Metadata payloads loaded from train/eval/explain run directories."""

    training_metadata: TrainingMetadata
    evaluation_metadata: EvaluationMetadata
    explainability_metadata: ExplainabilityMetadata

@dataclass
class ThresholdComparisonResult:
    """Outcome of comparing evaluation metrics against promotion thresholds."""

    meets_thresholds: bool
    message: str
    target_sets: list[MetricSet]
    target_metrics: list[MetricName]
    directions: dict[MetricName, Direction]

@dataclass
class ProductionComparisonResult:
    """Outcome of comparing candidate model metrics against production."""

    beats_previous: bool
    message: str
    previous_production_metrics: dict | None

@dataclass
class PreviousProductionRunIdentity:
    """Identifiers for the currently registered production model run."""

    experiment_id: str | None
    train_run_id: str | None
    eval_run_id: str | None
    explain_run_id: str | None
    promotion_id: str | None

EPSILON = 1e-8

COMPARISON_DIRECTIONS = {
    Direction.MAXIMIZE: lambda new, old: new > old + EPSILON,
    Direction.MINIMIZE: lambda new, old: new < old - EPSILON
}
