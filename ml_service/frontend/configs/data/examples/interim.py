INTERIM_EXAMPLE = """data:
  name: example_dataset
  version: v3
  output:
    path_suffix: data.parquet
    format: parquet
    compression: snappy

data_schema:
  example_column_1: string
  example_column_2: integer
  example_column_3: float

raw_data_version: v1

cleaning:
  lowercase_columns: true
  strip_strings: true
  replace_spaces_in_columns: true
  replace_dashes_in_columns: true

invariants:
  example_column_2:
    allowed_values: [1, 2, 3]
  example_column_3:
    allowed_values: [0.0, 1.0, 2.0]

drop_duplicates: true
drop_missing_ints: true
min_rows: 100000

lineage:
  created_by: Name Surname
"""