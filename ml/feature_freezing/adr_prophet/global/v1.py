import pandas as pd
from pathlib import Path

# Configuration
TASK_NAME = "adr_prophet"
VERSION_NAME = "v1"
TARGET = "adr"
SEGMENT_NAME = "global"

# Load data
data = pd.read_parquet(Path("data/hotel_booking_optimized.parquet"))

# Construct datetime
# Map month names to numbers first
month_map = {
    'January': 1,'February': 2,'March': 3,'April': 4,'May': 5,'June': 6,
    'July': 7,'August': 8,'September': 9,'October': 10,'November': 11,'December': 12
}

# Create a temporary DataFrame for to_datetime
df_dates = pd.DataFrame({
    "year": data["arrival_date_year"],
    "month": data["arrival_date_month"].map(month_map),
    "day": data["arrival_date_day_of_month"]
})

# Convert to datetime
data["arrival_date"] = pd.to_datetime(df_dates)

# Prepare Prophet dataframe
df_prophet = data[["arrival_date", TARGET]].rename(columns={
    "arrival_date": "ds",
    TARGET: "y"
})

# Sort by date
df_prophet = df_prophet.sort_values("ds")

# Split train/val/test by time
# Reserve last 20% of days for test, 15% for val
n = len(df_prophet)
test_size = int(n * 0.2)
val_size = int(n * 0.15)

df_train = df_prophet.iloc[:-(test_size + val_size)]
df_val = df_prophet.iloc[-(test_size + val_size):-test_size]
df_test = df_prophet.iloc[-test_size:]

# Save to files
feature_path = Path(f"data/features/{TASK_NAME}/{SEGMENT_NAME}/{VERSION_NAME}/")
feature_path.mkdir(parents=True, exist_ok=True)

df_train.to_parquet(feature_path / "df_train.parquet", index=False)
df_val.to_parquet(feature_path / "df_val.parquet", index=False)
df_test.to_parquet(feature_path / "df_test.parquet", index=False)

# Save schema
schema = pd.DataFrame({
    "feature": df_train.columns,
    "dtype": df_train.dtypes.astype(str)
})
schema.to_csv(feature_path / "schema.csv", index=False)

print(f"Global ADR Prophet features saved to {feature_path}")