import logging

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def validate_min_rows_after_cleaning(df, min_rows):
    if len(df) < min_rows:
        msg = f"Dataframe has only {len(df)} rows after cleaning, which is less than the minimum required {min_rows} rows."
        logger.error(msg)
        raise DataError(msg)