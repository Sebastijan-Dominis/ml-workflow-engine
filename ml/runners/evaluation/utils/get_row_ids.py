import logging

import pandas as pd

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def get_row_ids(dataset: pd.DataFrame) -> pd.Series:
    if "row_id" not in dataset.columns:
        msg = "The dataset does not contain a 'row_id' column. Please ensure the dataset includes a unique identifier for each row."
        logger.error(msg)
        raise DataError(msg)
    
    return dataset["row_id"]