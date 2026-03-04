"""Validation utilities for classification and regression target constraints."""

import logging

import pandas as pd
from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import ConfigError, DataError
from ml.features.validation.normalize_dtype import normalize_dtype

logger = logging.getLogger(__name__)

def validate_min_class_count(y: pd.Series, min_class_count: int):
    """Validate minimum sample count requirements across target classes.

    Args:
        y: Target class labels.
        min_class_count: Minimum count required per class.

    Returns:
        None: Raises on validation failure.
    """

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
    model_cfg: TrainModelConfig | SearchModelConfig,
    data: pd.DataFrame
) -> None:
    """Validate target nullability, dtype, and task-specific target constraints.

    Args:
        y: Target series.
        model_cfg: Model configuration with target constraints.
        data: Full data frame used for additional class-count checks.

    Returns:
        None: Raises on validation failure.
    """

    if y.isnull().any():
        msg = "Target variable contains null values."
        logger.error(msg)
        raise DataError(msg)

    actual_dtype = normalize_dtype(y.dtype)
    allowed = model_cfg.target.allowed_dtypes
    if actual_dtype not in allowed:
        msg = f"Target variable has dtype {y.dtype}, expected one of {allowed}."
        logger.error(msg)
        raise DataError(msg)

    if model_cfg.task.type == "classification":
        if model_cfg.target.classes is None:
            msg = "Classes configuration must be provided for classification problems."
            logger.error(msg)
            raise ConfigError(msg)
        positive_class = model_cfg.target.classes.positive_class
        if positive_class not in y.unique():
            msg = f"Positive class {positive_class} not found in target variable."
            logger.error(msg)
            raise DataError(msg)
        validate_min_class_count(
            data[model_cfg.target.name],
            model_cfg.target.classes.min_class_count
        )
        return # No further checks for classification

    target_constraints = model_cfg.target.constraints
    min_val = target_constraints.min_value
    max_val = target_constraints.max_value
    if model_cfg.task.type == "regression":
        if min_val is None and max_val is None:
            logger.warning("Min and max value constraints are not set for regression problem.")
        elif min_val is None:
            logger.warning("Min value constraint is not set for regression problem.")
        elif max_val is None:
            logger.warning("Max value constraint is not set for regression problem.")
    if min_val is not None and y.min() < min_val:
        msg = f"Target min {y.min()} < allowed min {min_val}"
        logger.error(msg)
        raise DataError(msg)
    if max_val is not None and y.max() > max_val:
        msg = f"Target max {y.max()} > allowed max {max_val}"
        logger.error(msg)
        raise DataError(msg)
    logger.debug("Target validation passed.")
