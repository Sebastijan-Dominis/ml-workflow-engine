"""Module for assessing model performance in post-promotion monitoring."""
import argparse
import logging
from typing import Any, Literal

import pandas as pd

from ml.exceptions import ConfigError, PipelineContractError
from ml.post_promotion.monitoring.extraction.expected_performance import get_expected_performance
from ml.post_promotion.monitoring.performance.calculation import calculate_current_performance
from ml.promotion.config.promotion_thresholds import MetricName, PromotionMetricsConfig
from ml.promotion.config.registry_entry import RegistryEntry

logger = logging.getLogger(__name__)

def assess_model_performance(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"],
    target: pd.Series,
    promotion_metrics_info: PromotionMetricsConfig
) -> dict[str | MetricName, dict[str, Any]]:
    """Assess model performance against expected thresholds.

    Args:
        args: Command-line arguments containing necessary identifiers.
        model_metadata: Metadata for the model being evaluated.
        stage: The stage of the model being evaluated ("production" or "staging").
        target: The target variable for the inference data.
        promotion_metrics_info: Configuration containing expected performance thresholds and directions for each metric.

    Returns:
        A dictionary containing the performance assessment results for each metric.
    """
    expected_performance = get_expected_performance(model_metadata, promotion_metrics_info)
    current_performance = calculate_current_performance(
        args=args,
        model_metadata=model_metadata,
        stage=stage,
        target=target
    )

    performance_results: dict[str | MetricName, dict[str, Any]] = {}

    for metric in promotion_metrics_info.metrics:
        expected_value = expected_performance[metric]

        current_value = current_performance[metric]
        if current_value is None or not isinstance(current_value, (int, float)):
            msg = f"Current performance is missing value for metric '{metric}'. Cannot assess model performance against expected thresholds. Current performance content: {current_performance}"
            logger.error(msg)
            raise PipelineContractError(msg)

        direction = promotion_metrics_info.directions[metric]
        if direction == "maximize":
            if current_value < expected_value:
                logger.warning(f"{stage.capitalize()} model performance degradation detected for metric '{metric}': expected >= {expected_value:.4f}, got {current_value:.4f}")
                performance_results[metric] = {
                    "status": "degradation",
                    "expected": expected_value,
                    "current": current_value,
                    "direction": direction
                }
            else:
                logger.info(f"{stage.capitalize()} model performance for metric '{metric}' is acceptable: expected >= {expected_value:.4f}, got {current_value:.4f}")
                performance_results[metric] = {
                    "status": "acceptable",
                    "expected": expected_value,
                    "current": current_value,
                    "direction": direction
                }
        elif direction == "minimize":
            if current_value > expected_value:
                logger.warning(f"{stage.capitalize()} model performance degradation detected for metric '{metric}': expected <= {expected_value:.4f}, got {current_value:.4f}")
                performance_results[metric] = {
                    "status": "degradation",
                    "expected": expected_value,
                    "current": current_value,
                    "direction": direction
                }
            else:
                logger.info(f"{stage.capitalize()} model performance for metric '{metric}' is acceptable: expected <= {expected_value:.4f}, got {current_value:.4f}")
                performance_results[metric] = {
                    "status": "acceptable",
                    "expected": expected_value,
                    "current": current_value,
                    "direction": direction
                }
        else:
            msg = f"Invalid direction '{direction}' for metric '{metric}'. Must be 'higher' or 'lower'."
            logger.error(msg)
            raise ConfigError(msg)
    return performance_results
