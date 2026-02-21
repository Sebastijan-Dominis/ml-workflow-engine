import logging
from typing import Optional

from ml.exceptions import ConfigError, UserError
from ml.promotion.config.models import Direction, MetricName, MetricSet
from ml.promotion.constants.constants import (COMPARISON_DIRECTIONS, Direction,
                                              MetricName, MetricSet,
                                              ProductionComparisonResult)

logger = logging.getLogger(__name__)

def compare_against_production_model(
    *,
    evaluation_metrics: dict[str, dict[str, float]], 
    current_prod_model_info: Optional[dict],
    metric_sets: list[MetricSet],
    metric_names: list[MetricName],
    directions: dict[MetricName, Direction]
) -> ProductionComparisonResult:
    if not current_prod_model_info:
        msg = "No current production model found. Skipping comparison against production model."
        logger.warning(msg)
        return ProductionComparisonResult(
            beats_previous=True,
            message=msg,
            previous_production_metrics=None
        )
    
    prod_metrics = current_prod_model_info.get("metrics", {})
    if not prod_metrics:
        msg = "Current production model does not have metrics information."
        logger.error(msg)
        raise UserError(msg)
    
    for metric_set in metric_sets:
        for metric in metric_names:
            prod_metric_value = prod_metrics.get(metric_set, {}).get(metric)
            if prod_metric_value is None:
                msg = f"Production model is missing metric '{metric}' in set '{metric_set}'. Cannot compare against production model."
                logger.error(msg)
                raise UserError(msg)

            eval_metric_value = evaluation_metrics.get(metric_set, {}).get(metric)
            if eval_metric_value is None:
                msg = f"Evaluation metrics are missing metric '{metric}' in set '{metric_set}'. Cannot compare against production model."
                logger.error(msg)
                raise UserError(msg)

            direction = directions.get(metric)
            if direction is None:
                msg = f"Direction for metric '{metric}' is not defined."
                logger.error(msg)
                raise ConfigError(msg)
            comparison_func = COMPARISON_DIRECTIONS.get(direction)
            if not comparison_func:
                msg = f"Invalid direction '{direction}' for metric '{metric}'."
                logger.error(msg)
                raise ConfigError(msg)
            
            if not comparison_func(eval_metric_value, prod_metric_value):
                msg = f"Metric '{metric}' in set '{metric_set}' does not outperform production model. Evaluation value: {eval_metric_value}, Production value: {prod_metric_value}."
                logger.warning(f"Promotion criteria not met: {msg}")
                return ProductionComparisonResult(
                    beats_previous=False,
                    message=msg,
                    previous_production_metrics=prod_metrics
                )
            else:
                logger.debug(f"Metric '{metric}' in set '{metric_set}' outperforms production model. Evaluation value: {eval_metric_value}, Production value: {prod_metric_value}.")
    return ProductionComparisonResult(
        beats_previous=True,
        message="Model outperforms production model on all metrics.",
        previous_production_metrics=prod_metrics
    )