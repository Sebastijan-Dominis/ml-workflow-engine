"""Validation helpers for row identifier integrity in feature datasets."""

import logging

import pandas as pd
from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def validate_row_id(df: pd.DataFrame) -> None:
    """Validate that `row_id` exists and is unique within the dataframe.

    Args:
        df: Dataframe expected to contain unique ``row_id`` values.

    Returns:
        None.
    """

    if "row_id" not in df.columns:
        msg = "Each feature set must contain a 'row_id' column for proper merging and alignment with the target variable."
        logger.error(msg)
        raise DataError(msg)

    if df["row_id"].duplicated().any():
        dupes = df["row_id"][df["row_id"].duplicated()].unique()
        msg = f"row_id must be unique within each feature set. Found duplicates: {list(dupes)}."
        logger.error(msg)
        raise DataError(msg)

    logger.debug("Successfully validated that row_id column is present and contains unique values.")
