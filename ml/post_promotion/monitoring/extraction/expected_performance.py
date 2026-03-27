"""Module for extracting expected performance metrics from model metadata for post-promotion monitoring."""
import logging

from ml.exceptions import PipelineContractError
from ml.promotion.config.promotion_thresholds import PromotionMetricsConfig
from ml.promotion.config.registry_entry import RegistryEntry

logger = logging.getLogger(__name__)

def get_expected_performance(
    model_metadata: RegistryEntry,
    promotion_metrics_info: PromotionMetricsConfig
):
    """Extract expected performance metrics from model metadata for post-promotion monitoring."""
    expected_performance = {}

    for metric in promotion_metrics_info.metrics:
        curr_metric_value = model_metadata.metrics.test.get(metric, None)

        if not isinstance(curr_metric_value, (int, float)):
            msg = f"Model metadata has null expected value for metric '{metric}' in the test set. Cannot retrieve expected performance for monitoring comparison."
            logger.error(msg)
            raise PipelineContractError(msg)

        expected_performance[metric] = curr_metric_value

    return expected_performance
