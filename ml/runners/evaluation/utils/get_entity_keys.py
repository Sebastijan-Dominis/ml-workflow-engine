"""Helpers for extracting row identifiers from evaluation feature frames."""

import logging

import pandas as pd

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def get_entity_keys(data: pd.DataFrame, entity_key: str) -> pd.Series:
    """Return ``entity_key`` series from dataframe, raising if missing.

    Args:
        data: Input dataframe expected to contain a ``entity_key`` column.
        entity_key: The name of the entity key column to extract.

    Returns:
        Entity key series extracted from the dataframe.
    """

    if entity_key not in data.columns:
        msg = f"The data does not contain a '{entity_key}' column. Please ensure the data includes a unique identifier for each row."
        logger.error(msg)
        raise DataError(msg)

    return data[entity_key]
