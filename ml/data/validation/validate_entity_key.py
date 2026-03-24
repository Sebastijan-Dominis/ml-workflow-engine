"""Validation helpers for row identifier integrity in feature datasets."""

import logging

import pandas as pd

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def validate_entity_key(df: pd.DataFrame, entity_key: str) -> None:
    """Validate that `entity_key` exists and is unique within the dataframe.

    Args:
        df: Dataframe expected to contain unique ``entity_key`` values.

    Returns:
        None.
    """

    if entity_key not in df.columns:
        msg = "Each feature set must contain a 'entity_key' column for proper merging and alignment with the target variable."
        logger.error(msg)
        raise DataError(msg)

    if df[entity_key].duplicated().any():
        dupes = df[entity_key][df[entity_key].duplicated()].unique()
        msg = f"entity_key must be unique within each feature set. Found duplicates: {list(dupes)}."
        logger.error(msg)
        raise DataError(msg)

    logger.debug("Successfully validated that entity_key column is present and contains unique values.")
