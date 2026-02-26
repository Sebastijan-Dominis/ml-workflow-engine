import logging

import pandas as pd

from ml.exceptions import RuntimeMLException

logger = logging.getLogger(__name__)

def get_memory_usage(df: pd.DataFrame) -> float:
    """Calculate the memory usage of a DataFrame in megabytes."""
    try:
        return df.memory_usage(deep=True).sum() / (1024 * 1024)
    except Exception as e:
        msg = f"Error computing memory usage of the data. "
        logger.error(msg + f"Details: {str(e)}")
        raise RuntimeMLException(msg) from e