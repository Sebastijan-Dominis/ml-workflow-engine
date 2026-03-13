# Model Configurations

## Overview

Model configurations define the full specification required to run a machine - - - learning workflow, including:
- model specifications
- feature set selection
- hyperparameter search settings
- training parameters
- hardware execution settings
- lineage metadata

Configurations are validated using Pydantic schemas to ensure correctness and reproducibility.

Two configuration types exist:

| Type       | Purpose                                   |
| ---------- | ----------------------------------------- |
| **Search** | Defines hyperparameter search experiments |
| **Train**  | Defines deterministic model training runs |

Both configurations share the same model specification base but diverge in their execution settings.

## Configuration Lifecycle

A model configuration goes through the following steps:
1. Load YAML configuration
2. Resolve inheritance (extends)
3. Apply environment overlays
4. Merge best parameters (training only)
5. Validate configuration schema
6. Compute configuration hash

This ensures that all runs are:
- reproducible
- validated
- traceable

## Shared Model Specification

Both search and training configs inherit from a shared specification model.

Both follow the same nesting logic as model specs (model specifications).

Example:

```text
configs/search/cancellation/global/v1.yaml
configs/train/cancellation/global/v1.yaml
```


Core fields typically include:

| Field         | Description                              |
| ------------- | ---------------------------------------- |
| `name`        | Model identifier                         |
| `version`     | Version tag                              |
| `algorithm`   | Model algorithm (e.g., CatBoost)         |
| `task`        | Task type (classification or regression) |
| `feature_set` | Feature set used for training            |

These fields define the logical model definition independent of execution stage.

## Search Configuration

Search configurations define **hyperparameter exploration experiments**.

Schema: `SearchModelConfig`

Example:

```yaml
name: credit_default_model
version: v1

algorithm: catboost
task:
  type: classification

feature_set: credit_features_v2

extends:
  - shared_defaults

seed: 42
cv: 5
verbose: 100

search:
  random_state: 42

  broad:
    iterations: 200
    n_iter: 50

    param_distributions:
      model:
        depth: [4,6,8]
        learning_rate: [0.01,0.03,0.1]

  narrow:
    enabled: true
    iterations: 100
    n_iter: 20

search_lineage:
  created_by: sebastijan
  created_at: 2026-02-27T23:53:30Z
```

Key fields:

| Field            | Description                         |
| ---------------- | ----------------------------------- |
| `search`         | Hyperparameter search configuration |
| `seed`           | Random seed for reproducibility     |
| `cv`             | Cross-validation folds              |
| `verbose`        | Logging verbosity                   |
| `search_lineage` | Creation metadata                   |

## Training Configuration

Training configurations define **deterministic model training runs** using fixed hyperparameters.

Schema: `TrainModelConfig`

Example:

```yaml
name: credit_default_model
version: v1

algorithm: catboost
task:
  type: classification

feature_set: credit_features_v2

extends:
  - shared_defaults

seed: 42
cv: 5

training:
  iterations: 2000

  model:
    depth: 6
    learning_rate: 0.03

  ensemble:
    bagging_temperature: 1.0

  early_stopping_rounds: 50

training_lineage:
  created_by: sebastijan
  created_at: 2026-03-01T11:12:30Z
```

Key fields:

| Field              | Description              |
| ------------------ | ------------------------ |
| `training`         | Training hyperparameters |
| `seed`             | Random seed              |
| `cv`               | Cross-validation folds   |
| `training_lineage` | Creation metadata        |

Training configs often **inherit hyperparameters discovered during search**.

## Hyperparameter Search

Search configurations support **two-stage optimization**:

### Broad Search

Explores a wide hyperparameter space.

Example parameters:
- `depth`
- `learning_rate`
- `l2_leaf_reg`
- `min_data_in_leaf`

These are defined as candidate distributions.

### Narrow Search

Optional refinement around the best parameters discovered during the broad stage.

Parameters are defined using:

- offsets for integer parameters
- multiplicative factors for float parameters
- optional bounds

This allows automatic **local hyperparameter refinement**.

## Hardware Configuration

Both search and training workflows support configurable hardware execution.

Schema: `HardwareConfig`

Example:

```yaml
hardware:
  task_type: GPU
  devices: [0]
```

Supported backends:

| Task Type | Description   |
| --------- | ------------- |
| `CPU`     | CPU execution |
| `GPU`     | GPU execution |

Optional settings:
- `memory_limit_gb`
- `allow_growth`
- device selection

## Configuration Composition

Configs can be composed using `inheritance and overlays`.

### Config Inheritance

The `extends` field allows a configuration to inherit from one or more base configs.

```yaml
extends:
  - shared_defaults
  - catboost_defaults
```

Later configs override earlier ones.

### Environment Overlays

Environment-specific values can be applied during loading.

Example environments:

```text
configs/env/default.yaml
configs/env/local.yaml
configs/env/production.yaml
```

These overlays modify configuration values without changing the base config.

### Best Parameter Injection

During training, the system can automatically inject best parameters from a search run.

Source:
```text
search_run/metadata.json
```

These parameters are merged into the training configuration before validation.

## Configuration Hashing

After validation, a **SHA-256 hash** is computed for the configuration.

The hash:
- excludes _meta runtime metadata
- is computed from the normalized configuration payload
- uniquely identifies the model definition

Example metadata entry:

```text
_meta:
  config_hash: 4c3f0d6a...
```

This enables:
- experiment reproducibility
- model lineage tracking
- configuration auditing

## Validation

All model configurations are validated using **strict Pydantic schemas**.

Validation guarantees:
- unknown fields are rejected
- types are enforced
- required fields are present
- default values are applied

Invalid configurations raise a `ConfigError` during loading.