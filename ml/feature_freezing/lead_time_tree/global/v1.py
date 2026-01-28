import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from ml.components.feature_engineering.TotalStay import TotalStay

# Configuration
TASK_NAME = "lead_time_tree"
VERSION_NAME = "v1"
TARGET = "lead_time"
SEGMENT_NAME = "global"

LEAKY_COLUMNS = [
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "reservation_status_date",
    "booking_changes",
    "days_in_waiting_list",
    "total_of_special_requests",
    "assigned_room_type",
    "required_car_parking_spaces",
    "adr",
    "name",
    "email",
    "phone-number",
    "credit_card",
    "company",
]

USELESS_COLUMNS = ['is_canceled', 'lead_time', 'is_repeated_guest', 'reservation_status']

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

print(f"Global lead-time tree-based features saved to {feature_path}")
