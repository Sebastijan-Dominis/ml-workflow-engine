"""A module for comparing performance metrics between production and staging models in post-promotion monitoring."""

import logging
from typing import Any

from ml.exceptions import ConfigError, PipelineContractError
from ml.post_promotion.monitoring.classes.function_returns import MonitoringExecutionOutput
from ml.promotion.config.promotion_thresholds import MetricName

logger = logging.getLogger(__name__)

def compare_production_and_staging_performance(
    prod_monitoring_output: MonitoringExecutionOutput,
    stage_monitoring_output: MonitoringExecutionOutput
) -> dict[str, Any]:
    """Compare performance metrics between production and staging monitoring models.

    Args:
        prod_monitoring_output: The monitoring output from the production model.
        stage_monitoring_output: The monitoring output from the staging model.

    Returns:
        A dictionary containing the comparison results for each metric.
    """
    performance_comparison: dict[str | MetricName, dict[str, Any]] = {}

    for metric in prod_monitoring_output.performance_results:
        if metric not in stage_monitoring_output.performance_results:
            logger.warning(f"Metric '{metric}' is present in production monitoring results but missing in staging monitoring results. Skipping comparison for this metric.")
            continue
        prod_metric_info = prod_monitoring_output.performance_results[metric]
        stage_metric_info = stage_monitoring_output.performance_results[metric]

        prod_perf = prod_metric_info["current"]
        stage_perf = stage_metric_info["current"]

        if not isinstance(prod_perf, (int, float)) or not isinstance(stage_perf, (int, float)):
            msg = f"Current performance values for metric '{metric}' must be numeric for comparison. Got production: {prod_perf} ({type(prod_perf)}), staging: {stage_perf} ({type(stage_perf)})."
            logger.error(msg)
            raise PipelineContractError(msg)

        if prod_metric_info["direction"] != stage_metric_info["direction"]:
            msg = f"Direction for metric '{metric}' is different between production and staging: '{prod_metric_info['direction']}' vs '{stage_metric_info['direction']}'. This should not happen."
            logger.error(msg)
            raise ConfigError(msg)

        direction = prod_metric_info["direction"]
        if direction == "maximize":
            diff = stage_perf - prod_perf
            status = "improvement" if diff > 0 else "degradation" if diff < 0 else "no_change"
        elif direction == "minimize":
            diff = prod_perf - stage_perf
            status = "improvement" if diff > 0 else "degradation" if diff < 0 else "no_change"
        else:
            msg = f"Invalid direction '{direction}' for metric '{metric}'. Must be 'maximize' or 'minimize'."
            logger.error(msg)
            raise ConfigError(msg)

        performance_comparison[metric] = {
            "production": prod_perf,
            "staging": stage_perf,
            "difference": diff,
            "status": status,
            "direction": direction
        }

    return performance_comparison
