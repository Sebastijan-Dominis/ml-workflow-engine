"""Validation helpers ensuring row-id alignment between features and target."""

import logging

import pandas as pd
from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def validate_feature_target_row_id(X: pd.DataFrame, y_with_row_id: pd.DataFrame) -> None:
    """Validate that all feature `row_id` values are present in target data.

    Args:
        X: Feature dataframe containing ``row_id`` values.
        y_with_row_id: Target dataframe containing ``row_id`` values.

    Returns:
        None.
    """

    if 'row_id' not in X.columns:
        msg = "Feature set is missing 'row_id' column."
        logger.error(msg)
        raise DataError(msg)

    if 'row_id' not in y_with_row_id.columns:
        msg = "Target data is missing 'row_id' column."
        logger.error(msg)
        raise DataError(msg)

    feature_row_ids = set(X['row_id'])
    target_row_ids = set(y_with_row_id['row_id'])
    missing = feature_row_ids - target_row_ids

    if missing:
        msg = (
            f"Row ID mismatch between features and target. "
            f"Missing in target: {missing}."
        )
        logger.error(msg)
        raise DataError(msg)
    logger.debug("Feature-target row_id validation passed.")
