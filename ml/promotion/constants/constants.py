import logging
from dataclasses import dataclass
from typing import Literal

from ml.promotion.config.models import Direction, MetricName, MetricSet

logger = logging.getLogger(__name__)

Stage = Literal["staging", "production"]

@dataclass
class RunnersMetadata():
    train_metadata: dict
    eval_metadata: dict
    explain_metadata: dict

@dataclass
class ThresholdComparisonResult():
    meets_thresholds: bool
    message: str
    target_sets: list[MetricSet]
    target_metrics: list[MetricName]
    directions: dict[MetricName, Direction]

@dataclass
class ProductionComparisonResult():
    beats_previous: bool
    message: str
    previous_production_metrics: dict | None

@dataclass
class PreviousProductionRunIdentity():
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