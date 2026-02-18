import logging

import pandas as pd

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def make_arrival_date(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["arrival_date"] = pd.to_datetime(
            df["arrival_date_year"].astype(str)
            + "-"
            + df["arrival_date_month"].astype(str)
            + "-"
            + df["arrival_date_day_of_month"].astype(str),
            format="%Y-%B-%d",
            errors="raise"
        )
        return df
    
    except Exception as e:
        msg = f"Error creating 'arrival_date' column. "
        logger.error(msg + f"Details: {str(e)}")
        raise DataError(msg) from e