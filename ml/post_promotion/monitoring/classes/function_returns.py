"""A module for defining function return types in the monitoring pipeline."""
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml.promotion.config.promotion_thresholds import MetricName


@dataclass
class InferenceFeaturesAndTarget:
    """Data class to hold inference features and target for monitoring."""
    features: pd.DataFrame
    target: pd.Series

@dataclass
class MonitoringExecutionOutput:
    """Data class to hold the output of a monitoring execution."""
    drift_results: dict[str, float]
    performance_results: dict[str | MetricName, dict[str, Any]]
    model_version: str
