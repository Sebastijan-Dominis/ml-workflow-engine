import pandas as pd

from sklearn.model_selection import train_test_split
from pathlib import Path

# Configuration
TASK_NAME = "no_show"
VERSION_NAME = "v1"
TARGET = "no_show"
SEGMENT_NAME = "global"

LEAKY_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "name",
    "email",
    "phone-number",
    "credit_card",
    "company",
]

USELESS_COLUMNS = ['is_canceled', 'reserved_room_type']

# Load data
data = pd.read_parquet(Path("data/hotel_booking_optimized.parquet"))

# Create target
data[TARGET] = (data["reservation_status"] == "No-Show").astype(int)

# Prepare features and target
X = data.drop(LEAKY_COLUMNS + USELESS_COLUMNS + [TARGET], axis=1, errors="ignore")
y = data[TARGET].copy()

# Train / val / test split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
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

print(f"Global no-show features saved to {feature_path}")
