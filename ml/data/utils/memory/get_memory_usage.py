"""Memory usage helpers for pandas dataframes."""

import logging

import pandas as pd

from ml.exceptions import RuntimeMLError

logger = logging.getLogger(__name__)

def get_memory_usage(df: pd.DataFrame) -> float:
    """Calculate dataframe memory footprint in megabytes (deep mode).

    Args:
        df: Dataframe whose memory usage should be measured.

    Returns:
        Total dataframe memory usage in megabytes.
    """
    try:
        return df.memory_usage(deep=True).sum() / (1024 * 1024)
    except Exception as e:
        msg = f"Error computing memory usage of the data. "
        logger.error(msg + f"Details: {str(e)}")
        raise RuntimeMLError(msg) from e