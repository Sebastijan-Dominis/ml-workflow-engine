"""Validation helpers for minimum-row requirements in dataframes."""

import logging

import pandas as pd
from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def validate_min_rows(df: pd.DataFrame, min_rows: int):
    """Ensure dataframe row count meets configured minimum threshold.

    Args:
        df: Dataframe to validate.
        min_rows: Minimum required number of rows.

    Returns:
        None.
    """

    if not min_rows:
        logger.warning("Minimum rows constraint not set. Defaulting to 0, which means no minimum row requirement.")
        min_rows = 0

    logger.debug(f"Validating minimum rows: dataframe has {len(df)} rows, minimum required is {min_rows}.")

    if len(df) < min_rows:
        msg = f"Data has {len(df)} rows, which is less than the minimum required {min_rows} rows."
        logger.error(msg)
        raise DataError(msg)

    logger.debug(f"Minimum rows validation passed. Dataframe has {len(df)} rows, which meets the minimum requirement of {min_rows} rows.")
