# Processed Data Configurations

## 1. Location

Processed data configurations can be found at:

```text
configs/data/processed/{dataset_name}/{version}.yaml
```

- `dataset_name` represents the dataset these configurations apply to
- `version` represents the version of the configuration
- Example: 

```text
configs/data/processed/hotel_bookings/v1.yaml
```

These configurations are used by the pipeline:

```text
pipelines/data/build_processed_dataset.py
```

The purpose of the processed stage is to *transform interim datasets into modeling-ready datasets* by removing unnecessary columns and preparing the final dataset artifact.

## 2. Fields

### Top-Level Fields

| Field                  | Type         | Description                                  |
| ---------------------- | ------------ | -------------------------------------------- |
| `data`                 | object       | Dataset identity and output configuration    |
| `interim_data_version` | string       | Version of the interim dataset used as input |
| `remove_columns`       | list[string] | Columns to remove from the dataset           |
| `lineage`              | object       | Metadata describing config provenance        |

### `data`

Defines metadata for the processed dataset artifact.

| Field     | Type   | Description                           |
| --------- | ------ | ------------------------------------- |
| `name`    | string | Dataset name                          |
| `version` | string | Dataset version (`v{integer}` format) |
| `output`  | object | Output storage configuration          |

### `data.output`

Controls how the processed dataset is persisted.

| Field         | Type           | Description                                                   |
| ------------- | -------------- | ------------------------------------------------------------- |
| `path_suffix` | string         | Path suffix appended to the dataset location                  |
| `format`      | string         | Output format (currently only `parquet`)                      |
| `compression` | string or null | Compression codec (`snappy`, `gzip`, `brotli`, `lz4`, `zstd`) |


### `interim_data_version`

Specifies which interim dataset version should be used as the input.

The value must follow the version format:

```text
v{integer}
```

Valid example:

```yaml
interim_data_version: v2
```

Invalid examples:

```yaml
interim_data_version: 2
```
```yaml
interim_data_version: version2
```

```yaml
interim_data_version: interim_v2
```

If the version does not match the required format, the pipeline will raise a `ConfigError`.

### `remove_columns`

Specifies columns that should be removed from the dataset during processing.

These typically include:

- personally identifiable information (PII)
- raw identifiers not used for modeling
- columns unnecessary for downstream pipelines
- leaky columns

Example:

```yaml
remove_columns:
  - name
  - email
  - phone_number
  - credit_card
```

Columns are added or removed depending on modeling requirements.

### `lineage`

Stores metadata describing the origin of the configuration.

| Field        | Type     | Description                                  |
| ------------ | -------- | -------------------------------------------- |
| `created_by` | string   | Author of the configuration                  |
| `created_at` | datetime | Timestamp when the configuration was created |

Example:

```yaml
lineage:
  created_by: Sebastijan
  created_at: "2026-02-27T23:53:30Z"
```

This metadata helps maintain traceability of configuration changes.

## 3 Full Example

```yaml
data:
  name: hotel_bookings
  version: v1
  output:
    path_suffix: data.parquet
    format: parquet
    compression: snappy

interim_data_version: v1

remove_columns:
  - name
  - email
  - phone_number
  - credit_card

lineage:
  created_by: Sebastijan
  created_at: "2026-02-27T23:53:30Z"
```

## 4. Validation Guarantees

The processed configuration schema enforces:
- strict version formatting for `interim_data_version`
- valid dataset metadata
- structured lineage information
If validation fails, the pipeline raises a `ConfigError` (exit code `2`).