import logging

import pandas as pd

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def get_row_ids(data: pd.DataFrame) -> pd.Series:
    if "row_id" not in data.columns:
        msg = "The data does not contain a 'row_id' column. Please ensure the data includes a unique identifier for each row."
        logger.error(msg)
        raise DataError(msg)
    
    return data["row_id"]