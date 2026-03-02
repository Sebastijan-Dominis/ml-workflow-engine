"""Utilities for forward and inverse transformations of target variables."""

import logging

import numpy as np
import pandas as pd
from scipy.special import boxcox as scipy_boxcox
from scipy.special import inv_boxcox

from ml.config.validation_schemas.model_specs import TargetTransformConfig
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)


def transform_target(
    y: pd.Series,
    *,
    transform_config: TargetTransformConfig,
    split_name: str
) -> pd.Series:
    """Apply configured target transformation for a named dataset split.

    Args:
        y: Target values for a single split.
        transform_config: Target transformation configuration.
        split_name: Human-readable split label for logging.

    Returns:
        pd.Series: Transformed target values indexed like input `y`.
    """

    logger.debug(f"Running the transform_target function for the '{split_name}' split; y_min: {y.min()}, y_max: {y.max()}")
    if not transform_config.enabled:
        logger.debug("No target transformation enabled, skipping transform.")
        return y

    y_values = y.to_numpy()

    if transform_config.type == "log1p":
        logger.debug("Applying log1p transformation to target.")
        invalid_mask = y_values < -1  # log1p domain: x > -1
        if np.any(invalid_mask) or np.any(np.isnan(y_values)):
            logger.warning(
                "log1p transformation found invalid target values:\n%s",
                y_values[invalid_mask | np.isnan(y_values)]
            )
        try:
            transformed = np.log1p(y_values)
        except Exception as e:
            msg = f"Error applying log1p transformation to target."
            logger.exception(msg)
            raise ConfigError(msg) from e

    elif transform_config.type == "sqrt":
        if (y_values < 0).any():
            invalid_mask = y_values < 0
            msg = "Sqrt transformation requires non-negative target values. Invalid values:\n%s" % y_values[invalid_mask | np.isnan(y_values)]
            logger.error(msg)
            raise ConfigError(msg)
        logger.debug("Applying sqrt transformation to target.")
        transformed = np.sqrt(y_values)

    elif transform_config.type == "boxcox":
        if (y_values <= 0).any():
            invalid_mask = y_values <= 0
            msg = "Box-Cox transformation requires strictly positive target values. Invalid values:\n%s" % y_values[invalid_mask | np.isnan(y_values)]
            logger.error(msg)
            raise ConfigError(msg)

        if transform_config.lambda_value is None:
            msg = "Box-Cox transformation requires lambda_value."
            logger.error(msg)
            raise ConfigError(msg)

        logger.debug("Applying Box-Cox transformation to target with lambda=%.4f", transform_config.lambda_value)
        transformed = scipy_boxcox(y_values, transform_config.lambda_value)

    else:
        msg = f"Unsupported target transformation type: {transform_config.type}"
        logger.error(msg)
        raise ConfigError(msg)

    transformed_series = pd.Series(transformed, index=y.index, name=y.name)
    logger.info("Completed target transformation %s for the %s split. y_min: %s, y_max: %s", transform_config.type, split_name, transformed_series.min(), transformed_series.max())
    return transformed_series

def inverse_transform_target(
    y_transformed: np.ndarray,
    *,
    transform_config: TargetTransformConfig,
    split_name: str
) -> np.ndarray:
    """Apply inverse transformation to transformed target predictions or values.

    Args:
        y_transformed: Transformed target array.
        transform_config: Target transformation configuration.
        split_name: Human-readable split label for logging.

    Returns:
        np.ndarray: Values mapped back to the original target scale.
    """

    logger.debug(f"Running the inverse_transform_target function for the '{split_name}' split; y_transformed_min: {y_transformed.min()}, y_transformed_max: {y_transformed.max()}")
    if not transform_config.enabled:
        logger.debug("No target transformation enabled, skipping inverse transform.")
        return y_transformed

    if transform_config.type == "log1p":
        logger.debug("Applying inverse log1p transformation to target.")
        result = np.expm1(y_transformed)

    elif transform_config.type == "sqrt":
        logger.debug("Applying inverse sqrt transformation to target.")
        result = np.square(y_transformed)

    elif transform_config.type == "boxcox":
        if transform_config.lambda_value is None:
            msg = "Box-Cox inverse requires lambda_value."
            logger.error(msg)
            raise ConfigError(msg)

        logger.debug("Applying inverse Box-Cox transformation to target with lambda=%.4f", transform_config.lambda_value)
        result = inv_boxcox(y_transformed, transform_config.lambda_value)

    else:
        msg = f"Unsupported target transformation type: {transform_config.type}"
        logger.error(msg)
        raise ConfigError(msg)
    
    logger.info("Completed inverse target transformation '%s' for the '%s' split. y_min: %s, y_max: %s", transform_config.type, split_name, result.min(), result.max())
    return result