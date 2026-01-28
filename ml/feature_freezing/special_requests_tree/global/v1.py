import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
TASK_NAME = "special_requests_tree"
VERSION_NAME = "v1"
TARGET = "total_of_special_requests"
SEGMENT_NAME = "global"

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
       'arrival_date_day_of_month', 'is_repeated_guest', 'total_of_special_requests']

# Load data
data = pd.read_parquet(Path("data/hotel_booking_optimized.parquet"))

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

# Save schema
schema = pd.DataFrame({
    "feature": X_train.columns,
    "dtype": X_train.dtypes.astype(str)
})
schema.to_csv(feature_path / "schema.csv", index=False)

print(f"Global special-requests features saved to {feature_path}")
