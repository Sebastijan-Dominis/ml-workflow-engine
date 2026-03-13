# Model Specifications

This document describes the structure, fields, and usage of **model specification configurations**.

Model specifications define the core modeling contract for a machine learning problem.
They describe the target, task type, dataset segmentation, feature sources, algorithm, scoring policy, and other modeling constraints.

Model specs are **canonical configurations** used by both:
- hyperparameter search pipelines
- training pipelines

Search and train configurations reference a model specification to ensure that both workflows operate under the same modeling assumptions.

## 1. Location

Model specifications are defined in:
`configs/model_specs/{problem}/{segment}/{version}.yaml`

- **problem** — modeling problem identifier (e.g., `cancellation`)
- **segment** — dataset segment identifier (e.g., `global`)
- **version** — specification version (`v{integer}.yaml`)

Example:
`configs/model_specs/cancellation/global/v1.yaml`

## 2. Fields

### Top-Level Fields

| Field                 | Type    | Description                                      |
| --------------------- | ------- | ------------------------------------------------ |
| `problem`             | string  | Name of the modeling problem                     |
| `segment`             | object  | Dataset segment identifier                       |
| `version`             | string  | Model specification version (`v{integer}`)       |
| `task`                | object  | Task type configuration                          |
| `target`              | object  | Target variable definition                       |
| `segmentation`        | object  | Optional dataset filtering rules                 |
| `min_rows`            | integer | Minimum dataset size required to train the model |
| `split`               | object  | Dataset splitting configuration                  |
| `algorithm`           | string  | Algorithm family used for modeling               |
| `model_class`         | string  | Model class used during training                 |
| `pipeline`            | object  | Pipeline configuration reference                 |
| `scoring`             | object  | Metric scoring policy                            |
| `class_weighting`     | object  | Class imbalance handling configuration           |
| `feature_store`       | object  | Feature store dataset references                 |
| `explainability`      | object  | Post-training explainability settings            |
| `data_type`           | string  | Data modality (`tabular`, `time-series`)         |
| `model_specs_lineage` | object  | Configuration provenance metadata                |


### Problem and Segment

These fields define the **model identity** and **dataset segment**.

| Field                 | Type   | Description                  |
| --------------------- | ------ | ---------------------------- |
| `problem`             | string | Name of the modeling problem |
| `segment.name`        | string | Segment identifier           |
| `segment.description` | string | Optional segment description |

Example:

```yaml
problem: cancellation

segment:
  name: global
  description: All bookings
```

### Task Configuration

Defines the type of machine learning task.

| Field     | Type   | Description                                                          |
| --------- | ------ | -------------------------------------------------------------------- |
| `type`    | enum   | Task type (`classification`, `regression`, `ranking`, `time_series`) |
| `subtype` | string | Optional subtype (e.g., `binary`, `multiclass`)                      |

Example:

```yaml
task:
  type: classification
  subtype: binary
```

### Target Configuration

Defines the target variable and associated constraints.

| Field            | Type   | Description                          |
| ---------------- | ------ | ------------------------------------ |
| `name`           | string | Target column name                   |
| `version`        | string | Target definition version            |
| `allowed_dtypes` | list   | Allowed dataset dtypes               |
| `classes`        | object | Class metadata (classification only) |
| `constraints`    | object | Numeric target constraints           |
| `transform`      | object | Optional target transformation       |

Example:

```yaml
target:
  name: is_canceled
  version: v1
  allowed_dtypes:
    - int64
  classes:
    count: 2
    positive_class: 1
    min_class_count: 200
```

### Dataset Segmentation

Optional filters applied before training.

| Field              | Type    | Description                                        |
| ------------------ | ------- | -------------------------------------------------- |
| `enabled`          | boolean | Enables segmentation                               |
| `include_in_model` | boolean | Whether segmentation column remains in the dataset |
| `filters`          | list    | Filter rules applied to the dataset                |

Example:

```yaml
segmentation:
  enabled: false
```

### Data Splitting

Defines the strategy used to split datasets into train, validation, and test sets.

| Field          | Type    | Description                          |
| -------------- | ------- | ------------------------------------ |
| `strategy`     | string  | Splitting strategy (`random`)        |
| `stratify_by`  | string  | Column used for stratified sampling  |
| `test_size`    | float   | Fraction of data used for testing    |
| `val_size`     | float   | Fraction of data used for validation |
| `random_state` | integer | Random seed                          |

Example:

```yaml
split:
  strategy: random
  stratify_by: is_canceled
  test_size: 0.2
  val_size: 0.12
  random_state: 42
```

### Algorithm Configuration

Defines the algorithm family and implementation class.

| Field         | Type   | Description                      |
| ------------- | ------ | -------------------------------- |
| `algorithm`   | enum   | Algorithm family (`catboost`)    |
| `model_class` | string | Model class used during training |

Example:

```yaml
algorithm: catboost
model_class: CatBoostClassifier
```

### Pipeline Configuration

References the pipeline used during modeling.

| Field     | Type   | Description                    |
| --------- | ------ | ------------------------------ |
| `version` | string | Pipeline version               |
| `path`    | string | Path to pipeline configuration |

Example:

```yaml
pipeline:
  version: v1
  path: configs/pipelines/tabular/catboost/v1.yaml
```

### Feature Store

Defines the feature sets used during model training.

| Field          | Type   | Description             |
| -------------- | ------ | ----------------------- |
| `path`         | string | Root feature store path |
| `feature_sets` | list   | Feature set references  |

Example:

```yaml
feature_store:
  path: feature_store/
  feature_sets:
    - name: booking_context_features
      version: v1
      data_format: parquet
      file_name: features.parquet
```

### Scoring Policy

Defines which evaluation metric should be used.

| Field              | Type   | Description                         |
| ------------------ | ------ | ----------------------------------- |
| `policy`           | enum   | Scoring strategy                    |
| `fixed_metric`     | metric | Metric used when `policy=fixed`     |
| `pr_auc_threshold` | float  | Threshold used for adaptive scoring |

Example:

```yaml
scoring:
  policy: adaptive_binary
  pr_auc_threshold: 0.1
```

### Class Weighting

Controls class imbalance handling.

| Field                 | Type  | Description                                           |
| --------------------- | ----- | ----------------------------------------------------- |
| `policy`              | enum  | Weighting strategy (`off`, `if_imbalanced`, `always`) |
| `imbalance_threshold` | float | Threshold triggering weighting                        |
| `strategy`            | enum  | Weight calculation method                             |

Example

```yaml
class_weighting:
  policy: if_imbalanced
  imbalance_threshold: 0.1
  strategy: balanced
```

### Explainability

Controls post-training model explainability.

| Field     | Type    | Description                     |
| --------- | ------- | ------------------------------- |
| `enabled` | boolean | Enables explainability analysis |
| `top_k`   | integer | Number of features to report    |
| `methods` | object  | Explainability methods          |

Example:

```yaml
explainability:
  enabled: true
  top_k: 20
  methods:
    feature_importances:
      enabled: true
      type: PredictionValuesChange
    shap:
      enabled: true
      approximate: tree
```

### Data Type

Defines the modality of the model input.

Supported values:
- `tabular`
- `time-series` (planned)

Example:

```yaml
data_type: tabular
```

### Lineage

Tracks authorship and creation time.

| Field        | Type     | Description                 |
| ------------ | -------- | --------------------------- |
| `created_by` | string   | Author of the specification |
| `created_at` | datetime | ISO timestamp               |

Example:

```yaml
model_specs_lineage:
  created_by: Sebastijan
  created_at: "2026-02-27T23:53:30Z"
```

## 3. Validation Guarantees

Model specifications are validated using Pydantic schemas.

Validation ensures:
- target definitions match the task type
- classification tasks define class metadata
- regression tasks may optionally define target transforms
- class weighting is only used for classification
- scoring policies contain required parameters
- segmentation filters are consistent with segmentation state

Invalid configurations raise a `ConfigError`.