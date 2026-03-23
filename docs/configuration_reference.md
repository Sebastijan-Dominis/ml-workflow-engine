# Configuration Reference

This document explains where configuration files are located,
what fields they contain, and how they are used within the ML workflow.

## Overview

- Configurations are essential for orchestrating the ML workflow.
- The existing file structure should be followed, since the code depends on it.
    - This includes nesting and naming configurations appropriately.
> Note: Some of the current Pydantic models were largely adapted for the hotel_bookings dataset, but they are structured in a way that allows scalability if needed (if adding more datasets).

## Configuration Philosophy

The repository is designed around configuration-driven workflows.

Core design principles:

- Pipelines contain minimal hard-coded logic
- Behavior is controlled through versioned configuration files
- Each configuration change produces a new version
- Config resolution allows modular overrides across workflow stages

This approach ensures reproducibility, traceability, and safe experimentation.

## Configuration Resolution Order

- Modeling relies on configuration resolution during the search (hyperparameter) and train (model training) pipelines.
- The following diagram explains how configurations get resolved at runtime:

![Config Resolution Architecture Diagram](/assets/img/docs/architecture/config_resolution_v3.png)

- For more information on why this resolution logic 
was decided upon, please consult the [`decisions.md` document](architecture/decisions.md)
- For more information on the interaction of various 
configurations with pipelines and artifacts, please
consult [the architecture overview](architecture/overview.md)

## Versioning Rules

- All of the configurations, except global and algorithm-specific defaults, as well as environment configurations, are expected to be versioned.
- The non-versioned configurations can, but generally should not, be altered.
- Configurations are versioned through their names:
    - The format is `v{integer}.yaml`, e.g. `v1.yaml`.
    - This approach may seem fragile or non-semantic by some, but it works well within the expected scope.
    - Most configurations will likely not need more than a few versions.
    - This is because the configuration logic is modular.

## Configuration Types

Each document below describes a specific configuration group,
including its location, expected fields, validation logic, and usage examples.

| Configuration | Description |
|---------------|-------------|
| [Interim Data Configurations](configs/interim_data.md) | Data cleaning and validation |
| [Processed Data Configurations](configs/processed_data.md) | Feature-ready dataset creation |
| [Feature Registry](configs/feature_registry.md) | Feature set definitions |
| [Model Specifications](configs/model_specs.md) | Model specifications (canonical for model configurations) |
| [Model Configurations](configs/model_config.md) | Modeling-specific configurations (used by search and train pipelines) |
| [Pipelines Configurations](configs/pipelines.md) | Configurations for sklearn Pipelines |
| [Promotion Thresholds](configs/promotion.md) | Model promotion criteria and performance thresholds |

> Note: Defaults (global and algorithm-specific), as well as environment configurations, are covered within model configurations (`docs/configs/model_config.md`)