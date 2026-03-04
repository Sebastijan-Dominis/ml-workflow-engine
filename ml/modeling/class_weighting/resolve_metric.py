"""Resolve the scoring metric according to policy and dataset characteristics."""

import logging

from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import ConfigError
from ml.modeling.class_weighting.constants import SUPPORTED_SCORING_FUNCTIONS
from ml.modeling.class_weighting.models import DataStats

logger = logging.getLogger(__name__)

def resolve_metric(config: SearchModelConfig | TrainModelConfig, stats: DataStats | None) -> SUPPORTED_SCORING_FUNCTIONS:
    """Select and return the scoring metric for search/training workflows.

    Args:
        config: Validated training or search configuration with scoring policy.
        stats: Optional class-distribution statistics for adaptive policies.

    Returns:
        Scoring function name used by search/training routines.
    """

    policy = config.scoring.policy

    if policy == "fixed":
        if not config.scoring.fixed_metric:
            msg = "fixed_metric must be set in scoring config for fixed policy."
            logger.error(msg)
            raise ConfigError(msg)
        logger.info(f"Using fixed scoring metric: {config.scoring.fixed_metric}")
        return config.scoring.fixed_metric

    if policy == "regression_default":
        scoring = "neg_root_mean_squared_error" # == RMSE but works with sklearn's RandomizedSearchCV
        logger.info(f"Using default regression metric: {scoring}")
        return scoring

    if stats is None:
        msg = f"Stats must be provided for non-fixed scoring policies. Got None for policy {policy}."
        logger.error(msg)
        raise ConfigError(msg)

    if policy == "adaptive_binary":
        if not config.scoring.pr_auc_threshold:
            msg = "pr_auc_threshold must be set in scoring config for adaptive_binary policy."
            logger.error(msg)
            raise ConfigError(msg)
        if stats.minority_ratio < config.scoring.pr_auc_threshold:
            scoring = "average_precision" # average_precision is basically the same as pr_auc, but it uses a different method to calculate the area under the curve. It calculates the area using a step function. The terms are used interchangeably in this project.
            logger.info(f"Minority ratio {stats.minority_ratio:.4f} is below threshold {config.scoring.pr_auc_threshold:.4f}, using PR-AUC.")
            return scoring
        scoring = "roc_auc"
        logger.info(f"Minority ratio {stats.minority_ratio:.4f} is above threshold {config.scoring.pr_auc_threshold:.4f}, using ROC-AUC.")
        return scoring

    msg = f"Unsupported scoring policy: {policy}"
    logger.error(msg)
    raise ConfigError(msg)
