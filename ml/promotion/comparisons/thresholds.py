import logging

from ml.exceptions import ConfigError, UserError
from ml.promotion.config.models import PromotionThresholds
from ml.promotion.constants.constants import (COMPARISON_DIRECTIONS,
                                              ThresholdComparisonResult)

logger = logging.getLogger(__name__)

def compare_against_thresholds(
    *,
    evaluation_metrics: dict[str, dict[str, float]], 
    promotion_thresholds: PromotionThresholds
) -> ThresholdComparisonResult:
    target_sets = promotion_thresholds.promotion_metrics.sets
    target_metrics = promotion_thresholds.promotion_metrics.metrics
    directions = promotion_thresholds.promotion_metrics.directions

    for metric_set in target_sets:
        thresholds_for_set = promotion_thresholds.thresholds.model_dump().get(metric_set)
        if thresholds_for_set is None:
            msg = f"Thresholds for metric set '{metric_set}' are not defined."
            logger.error(msg)
            raise ConfigError(msg)
        
        for metric in target_metrics:
            threshold_value = thresholds_for_set.get(metric)
            if threshold_value is None:
                msg = f"Threshold value for metric '{metric}' in set '{metric_set}' is not defined."
                logger.error(msg)
                raise ConfigError(msg)
            
            metric_value = evaluation_metrics.get(metric_set, {}).get(metric)
            if metric_value is None:
                msg = f"Evaluation metric '{metric}' is not available in the evaluation metrics."
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
            
            if not comparison_func(metric_value, threshold_value):
                msg = f"Metric '{metric}' with value {metric_value} does not meet the promotion threshold of {threshold_value} in set '{metric_set}'."
                logger.warning(f"Promotion criteria not met: {msg}")
                return ThresholdComparisonResult(
                    meets_thresholds=False,
                    message=msg,
                    target_sets=target_sets,
                    target_metrics=target_metrics,
                    directions=directions
                )
            else:
                logger.debug(f"Metric '{metric}' with value {metric_value} meets the promotion threshold of {threshold_value} in set '{metric_set}'.")
    return ThresholdComparisonResult(
        meets_thresholds=True,
        message="All promotion criteria regarding thresholds met.",
        target_sets=target_sets,
        target_metrics=target_metrics,
        directions=directions
    )
