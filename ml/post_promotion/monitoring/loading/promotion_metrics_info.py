import argparse
import logging
from pathlib import Path

from ml.exceptions import PipelineContractError
from ml.promotion.config.promotion_thresholds import PromotionMetricsConfig
from ml.promotion.validation.promotion_thresholds import validate_promotion_thresholds
from ml.utils.loaders import load_yaml

logger = logging.getLogger(__name__)


def get_promotion_metrics_info(
    args: argparse.Namespace,
) -> PromotionMetricsConfig:
    """Load and validate promotion metrics information for post-promotion monitoring.

    Args:
        args: Command-line arguments containing necessary identifiers.

    Returns:
        A PromotionMetricsConfig object containing the promotion metrics information.
    """
    global_thresholds = load_yaml(Path("configs") / "promotion" / "thresholds.yaml")
    model_thresholds_raw = global_thresholds.get(args.problem, {}).get(args.segment, {})
    if not model_thresholds_raw:
        msg = f"No promotion thresholds found for problem='{args.problem}' and segment='{args.segment}' in thresholds.yaml. File content: {global_thresholds}"
        logger.error(msg)
        raise PipelineContractError(msg)
    model_thresholds = validate_promotion_thresholds(model_thresholds_raw)
    return model_thresholds.promotion_metrics
