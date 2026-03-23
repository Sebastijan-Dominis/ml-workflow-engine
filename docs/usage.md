# Usage Guide

How to use the hotel_management project for typical workflows.

## Notes on orchestration

- **IMPORTANT**: All of the provided orchestrators should be used with caution, preferably not outside of development and testing (**major side-effects are possible**)
- Orchestrators were created almost exclusively for dev/test convenience and efficiency in single-owner, single-dev work
- `skip-if-existing` flag determines whether an orchestrator will run the given pipelines if it notices at least one snapshot in the target location (e.g. if set to true and `feature_store/{feature_set_name}/{feature_set_version}/{snapshot_id}/` already exists, it will not freeze that feature set; otherwise it will - new snapshot gets created)

## Docker

If you use docker, parts of the remainder of this file will be less relevant to you. Read regardless for more clarity.

For quick use with docker:

- Run the following command whenever the code is updated:

```bash
docker compose build
```

- Add `--no-cache` when environment changes:

```bash
docker compose build --no-cache
```

- Run this command to operate the ml workflow from your browser:

```bash
docker compose up
```

- Simply press `ctrl+c` stop running the container
    - Ideally avoid doing this while pipelines are mid-execution

- Backend is now on localhost:8000 by default
- Frontend is now on localhost:8050 by default

## ML Service

### Overview

- Code within the `ml_service/` folder provides `Dash` + `FastAPI` apps that can be used to:
    - Run pipelines
    - Create and store configs
        - Includes validation to ensure proper quality

#### Configurations

##### Supported

Currently *supports* the following configs:
- model specs + search + training
    - `configs/model_specs/{problem_type}/{segment}{model_version}.yaml`
    - `configs/search/{problem_type}/{segment}{model_version}.yaml`
    - `configs/train/{problem_type}/{segment}{model_version}.yaml`
- data (interim + processed)
    - `configs/data/interim/{dataset_name}/{dataset_version}.yaml`
    - `configs/data/processed/{dataset_name}/{dataset_version}.yaml`
- feature set configs (feature registry)
    - `configs/feature_registry/features.yaml`
- pipeline configs
    - `configs/pipelines/{data_type}/{algorithm}/{pipeline_version}.yaml`
- promotion thresholds
    - `configs/promotion/thresholds.yaml`

##### Not Supported

Currently *does not support* the following configs:
- defaults:
    - `configs/defaults/global.yaml`
    - `configs/defaults/{algorithm}.yaml`
- environment overlay (configs):
    - `configs/env/{env_name}.yaml`

##### Reasoning

- The supported defaults are written more often, require lineage with timestamp, and are versioned
- The unsupported configs are not meant to be altered, do not require lineage, and are not versioned

### Instructions

In order to use the ml service:

1. Launch the backend with

```bash
uvicorn ml_service.backend.main:app --reload
```

2. Launch the frontend with

```bash
python -m ml_service.frontend.app
```

3. Open the dashboard in your browser at the specified port and use it.

### Examples

#### Pipelines:

!["Gif portrayal of pipelines app from ml_service"](/assets/gifs/ml_service_pipelines_v2.gif)

#### Modeling Configs:

!["Gif portrayal of modeling configs app from ml_service"](/assets/gifs/ml_service_modeling_configs_v2.gif)

## Running Pipelines

- Use CLI commands with python scripts found in `pipelines/`
- This section includes a brief overview of the pipelines
- The diagrams describing each pipeline in more detail (e.g. which artifacts are used and produced at each step) can be viewed in [the architecture overview](architecture/overview.md)
- More architecture information in general is located in the `docs/architecture` folder as well
    - [decisions](architecture/decisions.md)
    - [system invariants](architecture/system_invariants.md)
    - [boundaries](architecture/boundaries.md)
    - [validation guarantees](architecture/validation_guarantees.md)

### Data Preprocessing

- The `pipelines/data/register_raw_snapshot.py` pipeline registers raw data it finds in `data/raw/{dataset_name}/{dataset_version}/{snapshot_id}/data.{format}`, based on cli arguments
- The `pipelines/data/build_interim_dataset.py` pipeline builds an interim dataset from one of the raw data snapshots, based on the interaction between cli arguments and configs from `configs/data/interim/{dataset_name}/{dataset_version}.yaml`
- The `pipelines/data/build_processed_dataset.py` pipeline builds a processed dataset from one of the interim datasets, based on the interaction between cli arguments and configs from `configs/data/processed/{dataset_name}/{dataset_version}.yaml`
- The `pipelines/orchestration/data/execute_all_data_preprocessing.py` orchestrator executes all of the three pipelines for all of the available raw snapshots and interim and processed configs

### Feature Set Freezing

- The `pipelines/features/freeze.py` pipelines freezes a feature set based on the interaction of cli arguments with feature registry (`configs/feature_registry/features.yaml`)
- The `pipelines/orchestration/features/freeze_all_feature_sets.py` orchestrator freezes all of the feature sets found in the feature registry (`configs/feature_registry/features.yaml`)

### Experiments

#### Search (hyperparameter searching)

- The `pipelines/search/search.py` pipeline performs a hyperparameter search for a given model, based on the interaction between cli arguments and resolved configs (a graph in [the architecture overview](architecture/overview.md) shows how configs are resolved at runtime)
- A *search* run *defines* an *experiment* (one search = one experiment)

#### Runners

##### Training

- The `pipelines/runners/train.py` pipeline performs a training run, based on the interaction between cli arguments and resolved configs (a graph in [the architecture overview](architecture/overview.md) shows how configs are resolved at runtime)
- A *training* run is *canonical* for *evaluation* and *explainability* runs

##### Evaluation

- The `pipelines/runners/evaluate.py` pipeline performs an evaluation run, based on cli arguments

##### Explainability

- The `pipelines/runners/explain.py` pipelines performs an explainability run, based on cli arguments

#### Orchestration

- The `pipelines/orchestration/experiments/execute_experiment_with_latest.py` orchestrator executes `search.py`, `train.py`, `evalute.py` and `explain.py` in sequence by defaulting to the latest experiment id for all runners, and the latest train run id for evaluation and explainability runs
- The `pipelines/orchestration/experiments/execute_all_experiments_with_latest.py` orchestrator executes `execute_experiment_with_latest.py` for all of the models, based on file structure within `configs/model_specs`, such that *problem type* + *segment* + *model version* = *one model*.

### Promotion

- The `pipelines/promotion/promote.py` stages or promotes a model, and archives the previous one (if promotion occurs), based on cli arguments and predefined thresholds (`configs/promotion/thresholds.yaml`)

### The Grand Orchestrator

- The `pipelines/orchestration/master/run_all_workflows.py` executes `execute_all_data_preprocessing.py`, `freeze_all_feature_sets.py` and `execute_all_experiments_with_latest.py` in sequence

## Artifacts

- All data-related artifacts can be found in `data/`
- All feature-set-related artifacts can be found in `feature_store/`
- All experiment-related artifacts can be found in `experiments/`
- All promotion-related artifacts, as well as the model registry and archive, can be found in `model_registry/`

## Scripts

- Use CLI commands with python scripts found in `scripts/`
- This section describes what each script does

### Generators

- The `generate_cols_for_row_id_fingerprint.py` script generates a fingerprint that ensures consistency in generating row_id of `hotel_bookings`
    - Impossible to ensure perfect consistency in python code alone, but acts as an additional sanity check
    - Good enough for local individual or small team use
- The `generate_fake_data.py` script generates fake data that can then be used by pipelines. 
    - The `data`, along with the `synthesizer_metadata` and a `quality_report`, is saved in 
    `data/raw/{dataset_name}/{dataset_version}/{dataset_snapshot}/`.
    - The trained model can be saved in `synthesizers/snapshot_id/`, named `ctgan_model.pkl` by default, and then reused, which greatly reduces
    the scripts' runtime (by up to 99% - training is expensive).
    - Data is stored in `csv` format by default. Alter the script if needs evolved.
    - The script is not modularized, as it is not considered to be a core part of the repo, and the repo comes with some pre-generated synthetic
    data, so the need for the script is not high.
    - May be modularized in the future.
    - The relationships between columns are likely not captured accurately, but that is considered acceptable at this stage.
        - Adding relationship logic would increase complexity with questionable justification for it.
        - The generated data is expected to be used for experimenting, rather than training production models.
    - This script requires extra setup steps to use, as mentioned in [setup.md](setup.md)
- The `generate_operator_hash.py` script generates an operator hash, which is needed when writing into the feature registry.
    - Ensure that the operators exist, and write them in proper format (e.g. TotalStay)
    - In CLI, separate the operators with a space character (e.g. --operators TotalStay ArrivalDate)
    - In GUI, use commas for separation (e.g. TotalStay, ArrivalDate)
- The `generate_snapshot_binding.py` script generates a new snapshot of snapshot bindings in the snapshot binding registry.
    - It always writes the latest snapshot for each existing dataset and feature set.
    - Alter the results manually if you need older snapshots for specific datasets and/or feature sets.

### Quality Scripts

- These scripts are used by the `pre-commit` hook, as well as `GitHub Actions CI`, to ensure code quality.
- The `check_import_layers.py` script checks import layers and dependencies across the codebase to enforce architectural boundaries (specified in [boundaries.md](architecture/boundaries.md))
- The `check_naming_conventions.py` script checks the naming conventions across the codebase. 
    - In order to satisfy the requirements:
        - use `snake_case` for `modules` and `functions`
        - use `PascalCase` for `classes`
        - do not prefix `module names` with `_` (except `__init__` and `__all__`)
    - The script also allows for ignoring certain folders, especially `tests/`

## Logging

- All individual data pipelines' logs can be found in `data/`
- All individual features pipelines' logs can be found in `feature_store/`
- All individual experiment-related pipelines' logs can be found in `experiments/`
- All individual promotion pipelines' logs can be found in `model_registry/`
- All orchestration logs can be found in `orchestration_logs/`
- Logging level is defined through CLI
- Expect detailed, informative logs from individual pipelines
- Expect high-level, helpful logs from orchestration pipelines
- Each pipeline run logs to a new location that is logically easy and intuitive to find
- All scripts' logs can be found in `scripts_logs/`

## EDA

- See `notebooks/EDA_and_Data_Preparation.ipynb` for initial exploration

## Configurations

- All configs are defined exclusively within `configs/`
- All configs have to respect the existing file structure - otherwise the pipelines will not work
- Naming of datasets, feature sets, versions, problem types and segments needs to be consistent across all configs
- All configs are currently required to be in `yaml`
- New change in anything that needs to be defined in configs -> new version
- The versioning format across the repository is v{integer} (e.g. v1, v2, v3), and it is important to respect this format in order for everything to work properly
- Whenever you see "version" in the context of configs within this repo, assume the v{integer} format

### Defaults

- Define global and algorithm-specific defaults in `configs/defaults/global.yaml` or `configs/defaults/{algorithm_name}.yaml`
- These configs are meant to be defined once and never changed
- The repo comes with some predefined configs - feel free to change them for your individual use-cases

### Data Configs

- Define configs for `build_interim_dataset.py` runs in `configs/data/interim/{dataset_name}/{dataset_version}.yaml`
- Define configs for `build_processed_dataset.py` runs in `configs/data/processed/{dataset_name}/{dataset_version}.yaml`

### Feature Registry

- Define configs for `freeze.py` in `configs/feature_registry/features.py`
- Generate the operator hash with `scripts/generators/generate_operator_hash.py`

### Model-specific Configs

- Every model needs exactly three configs defined - model specs, search and training
- Each of these should follow the exact same file structure in order for the pipelines to work as expected
- The expected nesting is `configs/{current}/{problem_type}/{segment}/{model_version}.yaml`, where current = `model_specs`, `search` or `training`
- Problem type can be cancellation, no_show, lead_time, etc.
- Segment can be global, city_hotel, resort_hotel_online_ta, etc. - use clear abbrevations
- Model specs are *foundational* for each model, while search and training configs help define what is relevant for search and training runs respectively (check [the architecture overview](architecture/overview.md) to understand how configs resolve at runtime)
- The repo comes with predefined model-specific configs for 14 models, spanning 7 problem types
- The predefined configs to not guarantee the most optimal results, but are considered to be a good starting point - adjust them as you wish

### Environment Configs

- Define environment configs in `configs/env/{environment}.yaml`, where environment = dev, prod or test
- The pipelines will only recognize dev, prod and test as valid names
- It is not crucial to have these configs defined, but they are useful, as they come last in config resolution
- Since they override all of the other configs, it is important to be mindful of what is included in these configs
- Their expected use is primarily convenience in dev/test environment and assurance of quality in prod environment
- The repo comes with some predefined configs for each of the environments - feel free to adjust them to your use-cases

### Pipeline Configs

- Define pipeline configs in `configs/pipelines/{data_type}/{algorithm}/{pipeline_version}.yaml`
- Data type can currently only be tabular; time-series is a planned implementation
- These configs define the logic used within an sklearn Pipeline that wraps some models
- Model specs define which pipeline version (if any) will be used

### Promotion Configs

- Define promotion thresholds for each model in `configs/promotion/thresholds.yaml`
- These configs are used by `promote.py`
- Thresholds can be changed, but changing them too often may be considered a bad business practice
- The repo comes with predefined thresholds for the 14 included models
- Thresholds are subjective, so you are encouraged to define them on your own