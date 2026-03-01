# These columns should *never* be changed, as they are used to create the row_id, which is the unique identifier for each row in the dataset. Changing any of these columns would change the row_id, which would break the lineage and tracking of data through the pipeline. If you need to change any of these columns, you must create a new column with the updated value and keep the original column for row_id creation. Ensure that each version of the dataset has the same values in these columns for the same rows to maintain consistent row_id values across versions, which is crucial for tracking and lineage.
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

COLS_FOR_ROW_ID_FINGERPRINT = "456cb2413ef2d2871406ae1e763ce4306e4fb1aad834a29e145cddb0f39bfbda"
