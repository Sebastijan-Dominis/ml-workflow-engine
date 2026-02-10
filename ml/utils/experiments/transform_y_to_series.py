import logging

import pandas as pd

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def transform_y_to_series(y: pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y_series = y.iloc[:, 0].copy()
        y_series.index = y.index
        return y_series
    if isinstance(y, pd.DataFrame) and y.shape[1] != 1:
        msg = f"Expected y to have a single column for binary classification, but got {y.shape[1]} columns."
        logger.error(msg)
        raise DataError(msg)
    if isinstance(y, pd.Series):
        return y
    msg = f"Expected y to be a DataFrame or Series, but got {type(y)}."
    logger.error(msg)
    raise DataError(msg)