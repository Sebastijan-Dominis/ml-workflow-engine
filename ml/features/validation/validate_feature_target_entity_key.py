"""Validation helpers ensuring row-id alignment between features and target."""

import logging

import pandas as pd

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def validate_feature_target_entity_key(X: pd.DataFrame, y_with_entity_key: pd.DataFrame, entity_key: str) -> None:
    """Validate that all feature `entity_key` values are present in target data.

    Args:
        X: Feature dataframe containing ``entity_key`` values.
        y_with_entity_key: Target dataframe containing ``entity_key`` values.
        entity_key: The name of the entity key column.

    Returns:
        None.
    """

    if entity_key not in X.columns:
        msg = f"Feature set is missing {entity_key} column."
        logger.error(msg)
        raise DataError(msg)

    if entity_key not in y_with_entity_key.columns:
        msg = f"Target data is missing {entity_key} column."
        logger.error(msg)
        raise DataError(msg)

    feature_entity_keys = set(X[entity_key])
    target_entity_keys = set(y_with_entity_key[entity_key])
    missing = feature_entity_keys - target_entity_keys

    if missing:
        msg = (
            f"Entity key mismatch between features and target. "
            f"Missing in target: {missing}."
        )
        logger.error(msg)
        raise DataError(msg)
    logger.debug("Feature-target entity_key validation passed.")
