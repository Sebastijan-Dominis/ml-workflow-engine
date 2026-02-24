import logging
from pathlib import Path

import pandas as pd

from ml.config.validation_schemas.model_cfg import (SearchModelConfig,
                                                    TrainModelConfig)
from ml.exceptions import ConfigError, DataError, PipelineContractError
from ml.utils.features.validation.normalize_dtype import normalize_dtype
from ml.config.validation_schemas.model_specs import TargetConfig
from ml.registry.hash_registry import hash_file
from ml.utils.features.hashing.hash_dataframe_content import hash_dataframe_content
from ml.utils.features.hashing.hash_feature_schema import hash_feature_schema

logger = logging.getLogger(__name__)

def ensure_required_fields_present(snapshot_path: Path, metadata: dict) -> None:
    required_fields = [
        "feature_schema_hash",
        "operators_hash",
        "feature_type",
        "loader_validation_hash",
        "in_memory_hash",
        "file_hash",
    ]

    missing_fields = [field for field in required_fields if field not in metadata]

    if missing_fields:
        msg = f"Metadata for snapshot {snapshot_path} is missing required fields: {', '.join(missing_fields)}"
        logger.error(msg)
        raise DataError(msg)

def validate_feature_set(
    feature_set: pd.DataFrame,
    *, 
    metadata: dict, 
    file_path: Path, 
    strict: bool = True
) -> None:
    if "row_id" not in feature_set.columns:
        msg = f"Feature set loaded from {file_path} is missing required 'row_id' column."
        logger.error(msg)
        raise DataError(msg)

    expected_schema_hash = metadata["feature_schema_hash"]

    actual_schema_hash = hash_feature_schema(feature_set)

    if actual_schema_hash != expected_schema_hash:
        msg = f"Feature schema hash mismatch: expected {expected_schema_hash}, got {actual_schema_hash}"
        logger.error(msg)
        raise DataError(msg)

    if strict:
        expected_in_memory_hash = metadata["in_memory_hash"]
        actual_in_memory_hash = hash_dataframe_content(feature_set)
        if actual_in_memory_hash != expected_in_memory_hash:
            msg = f"In-memory feature hash mismatch: expected {expected_in_memory_hash}, got {actual_in_memory_hash}"
            logger.warning(msg)

        expected_file_hash = metadata["file_hash"]
        actual_file_hash = hash_file(file_path)
        if actual_file_hash != expected_file_hash:
            msg = f"File hash mismatch: expected {expected_file_hash}, got {actual_file_hash}"
            logger.error(msg)
            raise DataError(msg)

def validate_set(hash_type: str, hashes: set, feature_sets: list) -> None:
    if len(hashes) != 1:
        msg = f"{hash_type} hashes do not match across feature sets. Feature sets involved: " + ", ".join(
            [f"{feature_sets[i].name} (version: {feature_sets[i].version})" for i in range(len(feature_sets))]
        )
        logger.error(msg)
        raise DataError(msg)
    
def validate_model_feature_pipeline_contract(model_cfg: SearchModelConfig | TrainModelConfig, pipeline_cfg: dict, cat_features: list | None = None) -> None:
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
        
def validate_min_class_count(y: pd.Series, min_class_count: int):
    if y.nunique() < 2:
        msg = "Target variable must have at least two classes for classification."
        logger.error(msg)
        raise DataError(msg)
    
    if not min_class_count:
        logger.warning("Minimum class count constraint not set.")
    
    logger.debug(f"Validating minimum class count: minimum required is {min_class_count}.")
    
    class_counts = y.value_counts()
    for cls, count in class_counts.items():
        if count < min_class_count:
            msg = f"Class {cls} has {count} instances, which is less than the minimum required {min_class_count}."
            logger.error(msg)
            raise DataError(msg)
        else:
            logger.debug(f"Class {cls} has {count} instances, which meets the minimum required {min_class_count}.")

def validate_target(
    *,
    y: pd.Series, 
    tgt_cfg: TargetConfig, 
    data: pd.DataFrame
) -> None:
    if y.isnull().any():
        msg = "Target variable contains null values."
        logger.error(msg)
        raise DataError(msg)
    
    actual_dtype = normalize_dtype(y.dtype)
    allowed = tgt_cfg.allowed_dtypes
    if actual_dtype not in allowed:
        msg = f"Target variable has dtype {y.dtype}, expected one of {allowed}."
        logger.error(msg)
        raise DataError(msg)
    
    if tgt_cfg.problem_type == "classification":
        if tgt_cfg.classes is None:
            msg = "Classes configuration must be provided for classification problems."
            logger.error(msg)
            raise ConfigError(msg)
        positive_class = tgt_cfg.classes.positive_class
        if positive_class not in y.unique():
            msg = f"Positive class {positive_class} not found in target variable."
            logger.error(msg)
            raise DataError(msg)
        validate_min_class_count(
            data[tgt_cfg.name],
            tgt_cfg.classes.min_class_count
        )
        return # No further checks for classification

    target_constraints = tgt_cfg.constraints
    min_val = target_constraints.min_value
    max_val = target_constraints.max_value
    if tgt_cfg.problem_type == "regression" and (min_val is None or max_val is None):
        logger.warning("Min and max value constraints are not set for regression problem.")
    if min_val is not None and y.min() < min_val:
        msg = f"Target min {y.min()} < allowed min {min_val}"
        logger.error(msg)
        raise DataError(msg)
    if max_val is not None and y.max() > max_val:
        msg = f"Target max {y.max()} > allowed max {max_val}"
        logger.error(msg)
        raise DataError(msg)
    