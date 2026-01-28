import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

from ml.components.feature_engineering.TotalStay import TotalStay

# Define function to save segment-specific features
def save_segment(segment):
    # Configuration
    TASK_NAME = "special_requests_tree"
    SEGMENT_NAME = segment["segment_name"].strip().replace(" ", "_").replace("/", "_").lower()
    VERSION_NAME = "v1"
    TARGET = "total_of_special_requests"
    SEGMENT_COLUMN_1 = "hotel"
    SEGMENT_COLUMN_2 = "customer_type"
    SEGMENT_VALUE_1 = segment["segment_value_1"]
    SEGMENT_VALUE_2 = segment["segment_value_2"]

    LEAKY_COLUMNS = [
        "booking_changes",
        "assigned_room_type",
        "required_car_parking_spaces",
        "days_in_waiting_list",
        "reservation_status",
        "reservation_status_date",
        "adr",
        "lead_time",
        "name",
        "email",
        "phone-number",
        "credit_card",
        "company",
    ]

    USELESS_COLUMNS = ['is_canceled', 'arrival_date_year', 'arrival_date_week_number',
       'arrival_date_day_of_month', 'is_repeated_guest', 'total_of_special_requests', SEGMENT_COLUMN_1, SEGMENT_COLUMN_2]

    # Load data
    data = pd.read_parquet(Path("data/hotel_booking_optimized.parquet"))

    # Create target
    data[TARGET] = (
        data["reserved_room_type"].astype(str)
        != data["assigned_room_type"].astype(str)
    ).astype(np.int8)

    # Apply segmentation
    HIGH_COUNT_CUSTOMER_TYPES = ["Transient", "Transient-Party"]

    if SEGMENT_VALUE_2.strip() == "Other":
        data = data[
            (data[SEGMENT_COLUMN_1].str.strip() == SEGMENT_VALUE_1.strip()) & 
            (~data[SEGMENT_COLUMN_2].str.strip().isin(HIGH_COUNT_CUSTOMER_TYPES))
        ].copy()
    else:
        data = data[
            (data[SEGMENT_COLUMN_1].str.strip() == SEGMENT_VALUE_1.strip()) & 
            (data[SEGMENT_COLUMN_2].str.strip() == SEGMENT_VALUE_2.strip())
        ].copy()

    # Prepare features and target
    X = data.drop(LEAKY_COLUMNS + USELESS_COLUMNS + [TARGET], axis=1, errors="ignore")
    y = data[TARGET].copy()

    # Train / val / test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42
    )

    # Save frozen features
    feature_path = Path(f"data/features/{TASK_NAME}/{SEGMENT_NAME}/{VERSION_NAME}/")
    feature_path.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(feature_path / "X_train.parquet", index=False)
    X_val.to_parquet(feature_path / "X_val.parquet", index=False)
    X_test.to_parquet(feature_path / "X_test.parquet", index=False)

    y_train.to_frame().to_parquet(feature_path / "y_train.parquet", index=False)
    y_val.to_frame().to_parquet(feature_path / "y_val.parquet", index=False)
    y_test.to_frame().to_parquet(feature_path / "y_test.parquet", index=False)

    # Save raw schema
    raw_schema = pd.DataFrame({
        "feature": X_train.columns,
        "dtype": X_train.dtypes.astype(str),
        "role": ["input" if not col == TARGET else "target" for col in X_train.columns],
    })
    raw_schema.to_csv(feature_path / "schema.csv", index=False)

    # Save derived schema
    X_sample = X_train.head(100)  # small sample to detect dtypes
    operators = [TotalStay()]

    derived_features = []

    for op in operators:
        X_sample = op.transform(X_sample)
        for f in op.output_features:
            derived_features.append({
                "feature": f,
                "dtype": str(X_sample[f].dtype),
                "role": "derived",
                "source_operator": op.__class__.__name__
            })

    derived_schema = pd.DataFrame(derived_features)
    derived_schema.to_csv(feature_path / "derived_schema.csv", index=False)

    print(f"{SEGMENT_VALUE_1} - {SEGMENT_VALUE_2} special requests features saved to {feature_path}")

def generate_features_for_segments():
    # Define model names, targets, and segment information
    segments = {
        # Count: 59404
        "segment_1": {
            "segment_name": "city_hotel_transient",
            "segment_value_1": "City Hotel",
            "segment_value_2": "Transient"
        },
        # Count: 17333
        "segment_2": {
            "segment_name": "city_hotel_transient_party",
            "segment_value_1": "City Hotel",
            "segment_value_2": "Transient-Party"
        },
        # Count: 2593
        "segment_3": {
            "segment_name": "city_hotel_other_customer_types",
            "segment_value_1": "City Hotel",
            "segment_value_2": "Other"
        },
        # Count: 30209
        "segment_4": {
            "segment_name": "resort_hotel_transient",
            "segment_value_1": "Resort Hotel",
            "segment_value_2": "Transient"
        },
        # Count: 7790
        "segment_5": {
            "segment_name": "resort_hotel_transient_party",
            "segment_value_1": "Resort Hotel",
            "segment_value_2": "Transient-Party"
        },
        # Count: 2060
        "segment_6": {
            "segment_name": "resort_hotel_other_customer_types",
            "segment_value_1": "Resort Hotel",
            "segment_value_2": "Other"
        },
    }

    for segment in segments.values():
        save_segment(segment)

generate_features_for_segments()