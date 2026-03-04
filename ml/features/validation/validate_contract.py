"""Validation helpers for model and feature-pipeline compatibility contracts."""

import logging

from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import PipelineContractError

logger = logging.getLogger(__name__)

def validate_model_feature_pipeline_contract(model_cfg: SearchModelConfig | TrainModelConfig, pipeline_cfg: dict, cat_features: list | None = None) -> None:
    """Validate task and categorical handling compatibility between model and pipeline.

    Args:
        model_cfg: Validated training or search model configuration.
        pipeline_cfg: Pipeline configuration dictionary.
        cat_features: Optional categorical feature list required by some algorithms.

    Returns:
        None.
    """

    pipeline_supported_tasks = []
    if pipeline_cfg.get("assumptions", {}).get("supports_classification"):
        pipeline_supported_tasks.append("classification")
    if pipeline_cfg.get("assumptions", {}).get("supports_regression"):
        pipeline_supported_tasks.append("regression")

    if model_cfg.task.type not in pipeline_supported_tasks:
        msg = f"Pipeline does not support the task type: {model_cfg.task.type}"
        logger.error(msg)
        raise PipelineContractError(msg)

    if model_cfg.algorithm == "catboost":
        if cat_features is None:
            msg = "Categorical features must be provided for CatBoost models."
            logger.error(msg)
            raise PipelineContractError(msg)

        if not pipeline_cfg.get("assumptions", {}).get("handles_categoricals", False):
            msg = "Pipeline does not support categorical features required by CatBoost."
            logger.error(msg)
            raise PipelineContractError(msg)
