# Interim Data Configurations

## 1. Location

Interim data configurations can found at:

```text
configs/data/interim/{dataset_name}/{version}.yaml
```
  
- `dataset_name` represents the name of the dataset they are intended for.
- `version` implies the version of configurations for that specific dataset.
- Example: 

```text
configs/data/interim/hotel_bookings/v1.yaml
```

These configurations are used by the pipeline:

```text
pipelines/data/build_interim_dataset.py
```

The purpose of the interim stage is to *transform raw data into memory-optimized interim datasets* by optimizing column data types.

## 2. Fields

### Top-level fields

|Field|Type|Description|
|:----|:---|:----------|
|`data`|object|Dataset identity and output configuration|
|`data_schema`|object|Expected schema of the interim dataset|
|`raw_data_version`|string|Raw dataset version used as input|
|`cleaning`|object|Column normalization rules|
|`invariants`|object|Column-level validation and filtering rules|
|`drop_duplicates`|bool|Whether duplicate rows should be removed|
|`drop_missing_ints`|bool|Whether rows with missing integer values should be dropped|
|`min_rows`|int|Minimum number of rows required after cleaning|
|`lineage`|object|Metadata describing config provenance|

### `data`

Defines metadata for the produced interim dataset.

|Field|Type|Description|
|:----|:---|:----------|
|`name`|string|Dataset name|
|`version`|string|Dataset version (`v{integer}` format)|
|`output`|object|Output storage configuration|

### `data.output`

Controls how the interim dataset is persisted.

|Field|Type|Description|
|:----|:---|:----------|
|`path_suffix`|string|Path suffix appended to the dataset location|
|`format`|string|Output format (currently only `parquet`)|
|`compression`|string or null|Compression codec (`snappy`, `gzip`, `brotli`, `lz4`, `zstd`)|

### `data_schema`

Defines the expected schema of the interim dataset.

Each field specifies the target dtype after preprocessing.

Example:
```yaml
data_schema:
    hotel: category
    lead_time: int16
    adr: float32
```

The pipeline will enforce these dtypes during preprocessing.

### `raw_data_version`

Specifies the raw dataset version used to build the interim dataset.

As previously mentioned, the value must follow the `v{integer}` format.

*Valid example*:
```yaml
raw_data_version: v2
```

*Invalid examples*:
```yaml
raw_data_version: 2
```

```yaml
raw_data_version: version2
```

```yaml
raw_data_version: raw_v2
```

### `cleaning`

Defines columns normalization rules applied before validation.

|Field|Type|Description|
|:----|:---|:----------|
|`lowercase_columns`|bool|Convert column names to lowercase|
|`strip_string`|bool|Remove leading/trailing whitespace from string columns|
|`replace_spaces_in_columns`|bool|Replace spaces in column names with `_`|
|replace_dashes_in_columns|bool|Replace dashes in column names with `_`|

These operations standardize raw data before schema validation.

### `invariants`

Defines column-level validation rules applied during preprocessing.

Each column can define:

|Field|Type|Description|
|:----|:---|:----------|
|`min`|object|Minimum allowed value
|`max`|object|Maximum allowed value
|`allowed_values`|list|Allowed categorical values

Example:
```yaml
invariants:
    lead_time:
        min:
            value: 0
            op: gte
```

### `BorderValue`

Used to define boundary checks.
Used by `min` and `max`.

|Field|Type|Description|
|:----|:---|:----------|
|`value`|float|Boundary value|
|`op`|string|Comparison operator

Supported operators:
```text
lt <
lte <=
gt >
gte >=
```

### Default Invariants

If a column does not specify invariants in the configuration:
- Default constraints from the internal registry will be applied automatically.
- These constraints are defined in:
    - `ml/policies/data/interim_constraints.py`
- These constraints can be extended when onboarding new datasets.

This ensures:
- invalid values cannot silently enter the pipeline
- invariant coverage is complete

*Row Filtering:*

### `drop_duplicates`

- Removes duplicate rows after cleaning.
- Default:
```yaml
drop_duplicates: true
```

### `drop_missing_ints`

- Drops rows containing missing values in integer columns
- Default:
```yaml
drop_missing_ints: true
```

### `min_rows`
- Minimum number of rows required after preprocessing.
- If the dataset contains fewer rows than this threshold, the pipeline will fail.

### `lineage`
Stores metadata describing the origin of the configuration.

|Field|Type|Description|
|:----|:---|:----------|
|`created_by`|string|Author of the config|
|`created_at`|datetime|Timestamp of config creation|

## 3. Full Example

```yaml
data:
  name: hotel_bookings
  version: v1
  output:
    path_suffix: data.parquet
    format: parquet
    compression: snappy

data_schema:
  hotel: category
  is_canceled: int8
  lead_time: int16
  arrival_date_year: int16
  arrival_date_month: category
  arrival_date_week_number: int8
  arrival_date_day_of_month: int8
  stays_in_weekend_nights: int8
  stays_in_week_nights: int8
  adults: int16
  children: int8
  babies: int8
  meal: category
  country: category
  market_segment: category
  distribution_channel: category
  is_repeated_guest: int8
  previous_cancellations: int8
  previous_bookings_not_canceled: int8
  reserved_room_type: category
  assigned_room_type: category
  booking_changes: int8
  deposit_type: category
  agent: string
  company: string
  days_in_waiting_list: int16
  customer_type: category
  adr: float32
  required_car_parking_spaces: int8
  total_of_special_requests: int8
  reservation_status: category
  reservation_status_date: datetime64[ns]
  name: string
  email: string
  phone_number: string
  credit_card: string

raw_data_version: v1

cleaning:
  lowercase_columns: true
  strip_strings: true
  replace_spaces_in_columns: true
  replace_dashes_in_columns: true

invariants:
  hotel:
    allowed_values: [Resort Hotel, City Hotel]
  is_canceled:
    allowed_values: [0, 1]
  lead_time:
    min:
      value: 0
      op: gte
  arrival_date_year:
    min:
      value: 2015
      op: gte
    max:
      value: 2017
      op: lte
  arrival_date_month:
    allowed_values: [January, February, March, April, May, June, July, August, September, October, November, December]
  arrival_date_week_number:
    min:
      value: 1
      op: gte
    max:
      value: 53
      op: lte
  arrival_date_day_of_month:
    min:
      value: 1
      op: gte
    max:
      value: 31
      op: lte
  stays_in_weekend_nights:
    min: 
      value: 0
      op: gte
  stays_in_week_nights:
    min:
      value: 0
      op: gte
  adults:
    min: 
      value: 0
      op: gte
  children:
    min: 
      value: 0
      op: gte
  babies:
    min: 
      value: 0
      op: gte
  meal:
    allowed_values: [BB, HB, FB, SC, Undefined]
  market_segment:
    allowed_values: ["Aviation", "Direct", "Complementary", "Corporate", "Groups", "Offline TA/TO", "Online TA", "Undefined"]
  distribution_channel:
    allowed_values: ["Direct", "Corporate", "GDS", "TA/TO", "Undefined"]
  is_repeated_guest:
    allowed_values: [0, 1]
  previous_cancellations:
    min:
      value: 0
      op: gte
  previous_bookings_not_canceled:
    min: 
      value: 0
      op: gte
  booking_changes:
    min: 
      value: 0
      op: gte
  deposit_type:
    allowed_values: [No Deposit, Non Refund, Refundable]
  days_in_waiting_list:
    min:
      value: 0
      op: gte
  adr:
    min: 
      value: 0
      op: gte
  required_car_parking_spaces:
    min:
      value: 0
      op: gte
  total_of_special_requests:
    min:
      value: 0
      op: gte
  reservation_status:
    allowed_values: [Canceled, Check-Out, No-Show]
  reserved_room_type:
    allowed_values: [A, B, C, D, E, F, G, H, L, P]
  assigned_room_type:
    allowed_values: [A, B, C, D, E, F, G, H, I, K, L, P]

drop_duplicates: true

drop_missing_ints: true

min_rows: 100000

lineage:
  created_by: Sebastijan
  created_at: "2026-02-27t23:53:30z"
```

## 4. Validation Guarantees

The interim configuration schema enforces:
- Strict dataset version format
- Invariant boundaries defined by registry policies
- Allowed categorical values
- Automatic default invariants for unspecified columns

If any validation rule fails, the pipeline raises a `ConfigError` (exit code `2`)
