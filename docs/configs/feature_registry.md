# Feature Registry Configuration

## 1. Location

Feature registry configurations are defined in:

```text
configs/feature_registry/features.yaml
```

Feature definitions are organized by feature set name and version:

```yaml
{feature_set_name}:
  {version}:
```

Example structure:

```yaml
booking_context_features:
  v1:
    ...
```

These configurations are used by the pipeline:

```text
pipelines/features/freeze.py
```

The pipeline materializes feature sets into the feature store, creating versioned snapshots that can be used by modeling workflows.

## 2. Feature Registry Overview

The feature registry defines:
- which datasets features are sourced from
- which columns belong to the feature set
- how features are categorized (roles)
- which feature engineering operators should run
- validation constraints for feature quality
- storage settings for frozen feature snapshots

Each feature set configuration produces immutable feature snapshots stored in the feature store.

## 3. Fields

### **Top-Level Fields**

| Field                | Type         | Description                                                |
| -------------------- | ------------ | ---------------------------------------------------------- |
| `type`               | string       | Feature set type (currently `tabular`)                     |
| `description`        | string       | Optional human-readable description                        |
| `data`               | list         | Source dataset definitions                                 |
| `min_rows`           | int          | Minimum number of rows required after feature construction |
| `feature_store_path` | path         | Destination path for frozen feature snapshots              |
| `columns`            | list[string] | Columns included in the feature set                        |
| `feature_roles`      | object       | Feature type classification                                |
| `operators`          | object       | Feature engineering operator configuration                 |
| `constraints`        | object       | Data quality constraints                                   |
| `storage`            | object       | Storage configuration for feature snapshots                |
| `lineage`            | object       | Metadata describing configuration provenance               |

### `type`

Specifies the feature set type.

Currently supported:

```text
tabular
```

Future implementations may include:

```text
time_series
```

### `data`

Defines the datasets used to build the feature set.

Multiple datasets can be specified and merged during feature freezing.

| Field         | Type   | Description                         |
| ------------- | ------ | ----------------------------------- |
| `ref`         | string | Base path reference for the dataset |
| `name`        | string | Dataset name                        |
| `version`     | string | Dataset version                     |
| `format`      | string | File format (`csv` or `parquet`)    |
| `merge_key`   | string | Key used to merge datasets          |
| `path_suffix` | string | Dataset file suffix                 |


Example:

```yaml
data:
  - ref: data/processed
    name: hotel_bookings
    version: v1
    format: parquet
    merge_key: row_id
    path_suffix: data.{format}
```

### `columns`

Defines the columns included in the feature set.

Example:

```yaml
columns:
  - hotel
  - lead_time
  - arrival_date_year
  - arrival_date_month
```

All columns must be assigned a role in `feature_roles`.

### `feature_roles`

Defines how features are categorized for downstream modeling.

| Field         | Type         | Description          |
| ------------- | ------------ | -------------------- |
| `categorical` | list[string] | Categorical features |
| `numerical`   | list[string] | Numerical features   |
| `datetime`    | list[string] | Datetime features    |

Example:

```yaml
feature_roles:
  categorical:
    - hotel
    - meal
  numerical:
    - lead_time
    - stays_in_week_nights
  datetime: []
```

Validation ensures that every column listed in columns appears exactly once across these role groups.

### `operators`

Defines feature engineering operators applied during feature freezing.

**Operator hash** is generated via a script found in `scripts/generators/generate_operator_hash.py`.

Operators can either:
- materialize features into the dataset
- define logical feature relationships

| Field               | Type         | Description                                           |
| ------------------- | ------------ | ----------------------------------------------------- |
| `mode`              | string       | Operator execution mode (`materialized` or `logical`) |
| `names`             | list[string] | Names of operators to execute                         |
| `hash`              | string       | Operator configuration hash used for reproducibility  |
| `required_features` | dict         | Mapping of operator names to required input features  |

Example:

```yaml
operators:
  mode: logical
  names:
    - TotalStay
    - ArrivalSeason
  hash: 2074c53b8d1e6124b67ba8e625315f86
  required_features:
    TotalStay: [stays_in_weekend_nights, stays_in_week_nights]
    ArrivalSeason: [arrival_date_week_number]
```

Validation ensures:
- operator names match keys in `required_features`
- all required features exist in the feature set

### `constraints`

Defines feature-level data quality checks.

| Field             | Type         | Description                                          |
| ----------------- | ------------ | ---------------------------------------------------- |
| `forbid_nulls`    | list[string] | Columns that must not contain null values            |
| `max_cardinality` | dict         | Maximum allowed cardinality for categorical features |

Example:

```yaml
constraints:
  forbid_nulls:
    - hotel
    - lead_time
  max_cardinality:
    hotel: 2
    arrival_date_month: 12
```

Validation ensures all constraint-referenced columns exist in the feature set.

### `storage`

Defines how frozen feature snapshots are stored.

| Field         | Type   | Description                         |
| ------------- | ------ | ----------------------------------- |
| `format`      | string | Snapshot storage format (`parquet`) |
| `compression` | string | Compression codec                   |

Example:

```yaml
storage:
  format: parquet
  compression: snappy
```

### `min_rows`

Specifies the minimum number of rows required after feature construction.

If the feature set contains fewer rows than this threshold, the pipeline fails.

### `feature_store_path`

Defines the path where feature snapshots will be stored.

Example:
```yaml
feature_store_path: feature_store/booking_context_features/v1
```

Each freeze operation produces a snapshot inside this location.

### `lineage`

Stores metadata describing the origin of the configuration.

| Field        | Type     | Description                         |
| ------------ | -------- | ----------------------------------- |
| `created_by` | string   | Author of the configuration         |
| `created_at` | datetime | Timestamp of configuration creation |

Example:

```yaml
lineage:
  created_by: Sebastijan
  created_at: "2026-02-27T23:53:30Z"
```

## 4. Full Example

```yaml
booking_context_features:
  v1:
    type: tabular
    description: Features describing the booking itself.

    data:
      - ref: data/processed
        name: hotel_bookings
        version: v1
        format: parquet
        merge_key: row_id
        path_suffix: data.{format}

    min_rows: 5000

    feature_store_path: feature_store/booking_context_features/v1

    columns:
      - hotel
      - lead_time
      - arrival_date_year

    feature_roles:
      categorical:
        - hotel
      numerical:
        - lead_time
        - arrival_date_year
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
      created_by: Sebastijan
      created_at: "2026-02-27T23:53:30Z"
```