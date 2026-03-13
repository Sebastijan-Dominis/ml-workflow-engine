# Pipeline Configurations

This document describes the structure, fields, and usage of pipeline configuration files used within the ML workflow. Pipeline configs define the steps, assumptions, and metadata for building `sklearn` pipelines.

## 1. Location

Pipeline configurations are defined in:
`configs/pipelines/{data_type}/{algorithm}/{pipeline_version}.yaml`

- `data_type`: currently only `tabular` (time-series support is planned).
- `algorithm`: name of the modeling algorithm (e.g., `catboost`).
- `pipeline_version`: version of the pipeline configuration (`v{integer}.yaml`).

Example:
- `configs/pipelines/tabular/catboost/v1.yaml`

## 2. Fields

**Top-Level Fields**

| Field         | Type            | Description                                                   |
| ------------- | --------------- | ------------------------------------------------------------- |
| `name`        | string          | Name of the pipeline                                          |
| `version`     | string          | Pipeline version (must follow `v{integer}` format)            |
| `description` | string or null  | Optional description of the pipeline                          |
| `steps`       | list of strings | Ordered list of pipeline steps                                |
| `assumptions` | object          | Assumptions about task compatibility and categorical handling |
| `lineage`     | object          | Metadata describing config provenance                         |

### `steps`
Pipeline steps define the sequence of operations executed in the pipeline. Each step maps to a component in the `PIPELINE_COMPONENTS` registry.

Current supported steps:
- SchemaValidator – validates input schema and feature types
- FillCategoricalMissing – fills missing values in categorical features
- FeatureEngineer – executes feature operators
- FeatureSelector – selects features based on model config
- Model – placeholder; model is injected separately

Example `steps` field:

```yaml
steps:
  - SchemaValidator
  - FillCategoricalMissing
  - FeatureEngineer
  - FeatureSelector
  - Model
```

### `assumptions`
Pipeline assumptions define supported data types and tasks. Required keys:
| Key                       | Type    | Description                                          |
| ------------------------- | ------- | ---------------------------------------------------- |
| `handles_categoricals`    | boolean | Whether the pipeline can handle categorical features |
| `supports_regression`     | boolean | Whether the pipeline supports regression tasks       |
| `supports_classification` | boolean | Whether the pipeline supports classification tasks   |

Example:

```yaml
assumptions:
  handles_categoricals: true
  supports_regression: true
  supports_classification: true
```

### `lineage`
Lineage tracks authorship and creation timestamp:

| Field        | Type     | Description                                    |
| ------------ | -------- | ---------------------------------------------- |
| `created_by` | string   | Name of the author of this config              |
| `created_at` | datetime | ISO-8601 timestamp when the config was created |


Example:

```yaml
lineage:
  created_by: Sebastijan
  created_at: "2026-02-27T23:53:30Z"
```

## 3. Validation Guarantees

Pipeline configurations are validated using Pydantic models. Validation ensures:

1. **Version format** – `v{integer}` (e.g., `v1`)
2. **Step correctness** – only known steps allowed (`SchemaValidator`, `FillCategoricalMissing`, `FeatureEngineer`, `FeatureSelector`, `Model`)
3. **Assumptions keys** – must include all required keys (`handles_categoricals`, `supports_regression`, `supports_classification`)
4. **Lineage presence** – config must include author and creation timestamp

Invalid configurations raise a `ConfigError`.

## 4. Full Example

```yaml
name: tabular_catboost_v1
version: v1
description: >
  A CatBoost pipeline for tabular data including schema validation, 
  missing value imputation, feature engineering, feature selection, 
  and a placeholder model step.

steps:
  - SchemaValidator
  - FillCategoricalMissing
  - FeatureEngineer
  - FeatureSelector
  - Model

assumptions:
  handles_categoricals: true
  supports_regression: true
  supports_classification: true

lineage:
  created_by: Sebastijan
  created_at: "2026-02-27T23:53:30Z"
```