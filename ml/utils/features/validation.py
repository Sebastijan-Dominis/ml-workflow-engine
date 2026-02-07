import logging
logger = logging.getLogger(__name__)
from pathlib import Path

from ml.exceptions import DataError, PipelineContractError
from ml.config.validation_schemas.model_cfg import SearchModelConfig

def validate_feature_set(snapshot_path: Path, metadata: dict) -> None:
    if not metadata["data_hash"]:
        msg = f"Invalid or missing data hash in metadata at {snapshot_path}"
        logger.error(msg)
        raise DataError(msg)

def validate_set(hash_type: str, hashes: set, feature_sets: list) -> None:
    if len(hashes) != 1:
        msg = f"{hash_type} hashes do not match across feature sets. Feature sets involved: " + ", ".join(
            [f"{feature_sets[i].name} (version: {feature_sets[i].version})" for i in range(len(feature_sets))]
        )
        logger.error(msg)
        raise DataError(msg)
    
def validate_model_feature_pipeline_contract(model_cfg: SearchModelConfig, pipeline_cfg: dict, cat_features: list | None = None) -> None:
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