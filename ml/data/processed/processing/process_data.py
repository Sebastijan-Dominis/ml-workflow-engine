import logging

import pandas as pd

from ml.data.processed.processing.new_cols.arrival_date import \
    make_arrival_date
from ml.data.utils.config.schemas.processed import ProcessedConfig
from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def remove_columns(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    missing_cols = [col for col in columns_to_remove if col not in df.columns]
    if missing_cols:
        msg = f"Cannot remove columns {missing_cols} because they are not present in the DataFrame."
        logger.error(msg)
        raise DataError(msg)
    
    return df.drop(columns=columns_to_remove)

def create_columns(df: pd.DataFrame, config: ProcessedConfig) -> pd.DataFrame:
    key_string = (
        df['hotel'].astype(str)
        + "_" + df['arrival_date_year'].astype(str)
        + "_" + df['arrival_date_month'].astype(str)
        + "_" + df['arrival_date_day_of_month'].astype(str)
        + "_" + df['reserved_room_type'].astype(str)
    )
    df['booking_id'] = pd.util.hash_pandas_object(key_string, index=False).astype(str)

    new_cols = config.create_columns
    if "arrival_date" in new_cols:
        df = make_arrival_date(df)
    return df