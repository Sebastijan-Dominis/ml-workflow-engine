EXAMPLE_CONFIG = """type: tabular
description: Example feature config

data:
  - ref: data/processed
    name: hotel_bookings
    version: v1
    format: parquet

min_rows: 5000

feature_store_path: feature_store/example_features/v1

columns:
  - hotel
  - lead_time

feature_roles:
  categorical:
    - hotel
  numerical:
    - lead_time
  datetime: []

constraints:
  forbid_nulls:
    - hotel
    - lead_time
  max_cardinality:
    hotel: 2

storage:
  format: parquet
  compression: snappy

lineage:
  created_by: Name Surname
"""
