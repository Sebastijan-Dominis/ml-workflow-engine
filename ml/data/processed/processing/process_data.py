import logging

import pandas as pd

from ml.exceptions import DataError
from ml.utils.data.validate_row_id import validate_row_id

logger = logging.getLogger(__name__)

cols_for_row_id = [
    'hotel',
    'arrival_date_year',
    'arrival_date_month',
    'arrival_date_day_of_month',
    'reserved_room_type',
    'assigned_room_type',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'meal',
    'market_segment',
    'distribution_channel',
    'is_repeated_guest',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'country',
    'agent',
]

def remove_columns(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    missing_cols = [col for col in columns_to_remove if col not in df.columns]
    if missing_cols:
        msg = f"Cannot remove columns {missing_cols} because they are not present in the DataFrame."
        logger.error(msg)
        raise DataError(msg)
    
    return df.drop(columns=columns_to_remove)

def add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns are present
    missing_cols = [col for col in cols_for_row_id if col not in df.columns]
    if missing_cols:
        msg = f"Cannot create row_id because required column(s) {missing_cols} are missing from the DataFrame."
        logger.error(msg)
        raise DataError(msg)

    # Sort by a stable key to ensure deterministic row_id assignment
    df = df.sort_values(by=cols_for_row_id).reset_index(drop=True)

    # Combine the columns into a string for each row
    key_string = df[cols_for_row_id].astype(str).agg('-'.join, axis=1)

    # Hash the string to get a unique row_id
    df['row_id'] = pd.util.hash_pandas_object(key_string, index=False).astype(str)

    # Handle exact duplicates by adding a duplicate counter
    dup_counts = df.groupby('row_id').cumcount()
    df['row_id'] = df['row_id'] + '-' + dup_counts.astype(str)

    validate_row_id(df)

    return df