PROCESSED_EXAMPLE = """data:
  name: example_dataset
  version: v4
  output:
    path_suffix: data.parquet
    format: parquet
    compression: snappy

interim_data_version: v2

remove_columns:
  - phone_number
  - credit_card

lineage:
  created_by: Name Surname
"""