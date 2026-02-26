import logging

from ml.config.validation_schemas.model_cfg import (SearchModelConfig,
                                                    TrainModelConfig)
from ml.exceptions import ConfigError
from ml.utils.experiments.class_weights.constants import \
    SUPPORTED_SCORING_FUNCTIONS
from ml.utils.experiments.class_weights.models import DataStats

logger = logging.getLogger(__name__)

def resolve_metric(config: SearchModelConfig | TrainModelConfig, stats: DataStats) -> SUPPORTED_SCORING_FUNCTIONS:
    policy = config.scoring.policy

    if policy == "fixed":
        if not config.scoring.fixed_metric:
            msg = "fixed_metric must be set in scoring config for fixed policy."
            logger.error(msg)
            raise ConfigError(msg)
        logger.info(f"Using fixed scoring metric: {config.scoring.fixed_metric}")
        return config.scoring.fixed_metric

    if policy == "regression_default":
        scoring = "rmse"
        logger.info(f"Using default regression metric: {scoring}")
        return scoring

    if policy == "adaptive_binary":
        if not config.scoring.pr_auc_threshold:
            msg = "pr_auc_threshold must be set in scoring config for adaptive_binary policy."
            logger.error(msg)
            raise ConfigError(msg)
        if stats.minority_ratio < config.scoring.pr_auc_threshold:
            scoring = "pr_auc"
            logger.info(f"Minority ratio {stats.minority_ratio:.4f} is below threshold {config.scoring.pr_auc_threshold:.4f}, using PR-AUC.")
            return scoring
        scoring = "roc_auc"
        logger.info(f"Minority ratio {stats.minority_ratio:.4f} is above threshold {config.scoring.pr_auc_threshold:.4f}, using ROC-AUC.")
        return scoring
    
    msg = f"Unsupported scoring policy: {policy}"
    logger.error(msg)
    raise ConfigError(msg)