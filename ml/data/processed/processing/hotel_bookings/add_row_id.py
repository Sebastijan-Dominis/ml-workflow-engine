import logging

import pandas as pd

from ml.data.processed.processing.add_row_id_base import AddRowIDBase
from ml.data.processed.processing.hotel_bookings.cols_for_row_id import (
    COLS_FOR_ROW_ID_FINGERPRINT, cols_for_row_id)
from ml.exceptions import DataError, UserError
from ml.utils.data.compute_cols_for_row_id_fingerprint import \
    compute_cols_for_row_id_fingerprint
from ml.utils.data.validate_row_id import validate_row_id

logger = logging.getLogger(__name__)

def validate_cols_for_row_id() -> str:
    expected_fingerpring = COLS_FOR_ROW_ID_FINGERPRINT
    actual_fingerprint = compute_cols_for_row_id_fingerprint(cols_for_row_id)
    
    if actual_fingerprint != expected_fingerpring:
        msg = (
            f"Cols for row_id have changed! This will change the row_id values and break lineage tracking. "
            f"Expected fingerprint: {expected_fingerpring}, Actual fingerprint: {actual_fingerprint}. "
            f"If you intentionally changed the cols_for_row_id, please update the COLS_FOR_ROW_ID_FINGERPRINT constant in cols_for_row_id.py with the new fingerprint."
        )
        logger.error(msg)
        raise UserError(msg)
    
    return actual_fingerprint

# This function could technically be silently changed, thus compromising the integrity of the row_id values and breaking lineage tracking. The current code structure does not allow for easily testing this function in isolation, but the fingerprint validation provides a way to detect unexpected changes in the code that generates row_id values, which is crucial for maintaining data integrity and lineage tracking. If the cols_for_row_id is changed, the fingerprint will change, causing the validation to fail and alerting developers to the change. This is especially important for hotel_bookings where row_id is used for tracking guests across datasets. In the future, third-party services can be used to ensure better traceability and monitoring of changes to this function, but for now, the fingerprint validation serves as a safeguard against unintended changes.
class AddRowIDToHotelBookings(AddRowIDBase):
    def add_row_id(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        fingerprint = validate_cols_for_row_id()

        # Ensure required columns are present
        missing_cols = [col for col in cols_for_row_id if col not in df.columns]
        if missing_cols:
            msg = f"Cannot create row_id because required column(s) {missing_cols} are missing from the DataFrame."
            logger.error(msg)
            raise DataError(msg)
        try:
            # Sort by a stable key to ensure deterministic row_id assignment
            df = df.sort_values(by=cols_for_row_id).reset_index(drop=True)

            # Combine the columns into a string for each row
            key_string = df[cols_for_row_id].astype(str).agg('-'.join, axis=1)

            # Hash the string to get a unique row_id
            df['row_id'] = pd.util.hash_pandas_object(key_string, index=False).astype(str)

            # Handle exact duplicates by adding a duplicate counter
            dup_counts = df.groupby('row_id').cumcount()
            df['row_id'] = df['row_id'] + '-' + dup_counts.astype(str)
        except Exception as e:
            msg = f"Failed to generate row_id: {e}"
            logger.error(msg)
            raise DataError(msg)

        validate_row_id(df)

        row_id_info = {
            "cols_for_row_id": cols_for_row_id,
            "fingerprint": fingerprint
        }

        return df, row_id_info