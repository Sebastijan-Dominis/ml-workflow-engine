"""Pydantic schemas and enums for promotion policy configuration."""

import logging
from datetime import datetime
from enum import StrEnum

from ml.exceptions import ConfigError
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

logger = logging.getLogger(__name__)

class MetricSet(StrEnum):
    """Supported metric split sets for promotion evaluation."""

    TEST = "test"
    VAL = "val"
    TRAIN = "train"

class MetricName(StrEnum):
    """Supported metric names used in promotion criteria."""

    ACCURACY = "accuracy"
    F1 = "f1"
    ROC_AUC = "roc_auc"

class Direction(StrEnum):
    """Optimization direction per promotion metric."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

class PromotionMetricsConfig(BaseModel):
    """Metric selection and direction configuration for promotion checks."""

    sets: list[MetricSet] = Field(..., description="List of metric sets to consider for promotion")
    metrics: list[MetricName] = Field(..., description="List of metrics to consider for promotion")
    directions: dict[MetricName, Direction] = Field(..., description="dictionary mapping each metric to its optimization direction")

    @field_validator("directions")
    @classmethod
    def validate_directions(
        cls,
        directions: dict[MetricName, Direction],
        info: ValidationInfo,
    ) -> dict[MetricName, Direction]:
        """Ensure direction entries are provided for every configured metric.

        Args:
            directions: Directions mapping.
            info: Pydantic validation context containing already parsed data.

        Returns:
            dict[MetricName, Direction]: Validated directions mapping.
        """

        metrics = set(info.data.get("metrics", []))
        direction_metrics = set(directions.keys())
        if metrics != direction_metrics:
            msg = (
                "Directions must be specified for all metrics. "
                f"Metrics: {metrics}, Directions provided for: {direction_metrics}"
            )
            logger.error(msg)
            raise ConfigError(msg)

        return directions

class ThresholdsConfig(BaseModel):
    """Per-split threshold values for promotion metrics."""

    test: dict[str, float] = Field(default_factory=dict, description="dictionary of metric thresholds for the test set")
    val: dict[str, float] = Field(default_factory=dict, description="dictionary of metric thresholds for the validation set")
    train: dict[str, float] = Field(default_factory=dict, description="dictionary of metric thresholds for the training set")

class LineageConfig(BaseModel):
    """Lineage metadata describing threshold config provenance."""

    created_by: str
    created_at: datetime

class PromotionThresholds(BaseModel):
    """Top-level validated promotion threshold configuration."""

    promotion_metrics: PromotionMetricsConfig
    thresholds: ThresholdsConfig
    lineage: LineageConfig

    @model_validator(mode="after")
    def validate_consistency(self):
        """Validate metric sets/metrics align with provided threshold blocks.

        Args:
            self: Candidate promotion thresholds instance.

        Returns:
            PromotionThresholds: Validated promotion-threshold object.
        """

        expected_sets = set(self.promotion_metrics.sets)
        thresholds_dump = self.thresholds.model_dump()

        actual_sets = {k for k, v in thresholds_dump.items() if v}
        if expected_sets != actual_sets:
            msg = f"Promotion metrics sets {expected_sets} do not match threshold sets {actual_sets}"
            logger.error(msg)
            raise ConfigError(msg)

        expected_metrics = set(self.promotion_metrics.metrics)

        for set_name in expected_sets:
            metrics_dict = thresholds_dump[set_name]
            actual_metrics = set(metrics_dict.keys())

            if expected_metrics != actual_metrics:
                msg = (
                    f"Promotion metrics {expected_metrics} do not match "
                    f"threshold metrics {actual_metrics} for set {set_name}"
                )
                logger.error(msg)
                raise ConfigError(msg)

        return self
