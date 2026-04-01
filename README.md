# ML Workflow Engine

## Table of Contents

- [Overview](#overview)
- [Why](#why)
- [Inspiration](#inspiration)
- [Key Achievements](#key-achievements)
- [Features](#features)
- [Example Use Case](#example-use-case)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Licence](#license)
- [Author](#author)
- [Contact](#contact)

## Overview

### An end-to-end ML platform that guarantees reproducibility across datasets, features, and models — with full lineage tracking and validation.

- Currently supports the modeling of regression and classification tasks using the CatBoost algorithm.
- Was initially formed based on a hotel booking dataset:
    - From: https://www.kaggle.com/datasets/mojtaba142/hotel-booking 
    - The architecture has since been expanded to support many datasets with minimal code changes.
- The ML workflow covers everything from the registration of a raw data snapshot to model monitoring.
- Designed with **production ML system constraints in mind**: reproducibility, traceability, validation, and modularity.
> Note: The repo was previously named `hotel_management`, so you will see that name around the repo; renamed for clarity 
> on what the project does.

> Another note: A few artifacts are intentionally included, along with their respective logs.
> This enables quick inspection of expected outputs of each pipeline, without having to run anything.

## Why?

1. Many ML platforms are either overengineered for small teams, or lack essential safeguards:
- For small teams, overengineering can be an issue:
  - Most small teams (1-5 developers) do not need worry about run conditions, very scalable storage, and so on
  - They need a simple, but strong and reliable platform
- Some teams fail in the other direction:
  - They fully rely on notebooks
  - They forget about validation and lineage tracking
  - They avoid elementary checks in order to "keep it simple"

This project keeps the workflow simple, while still providing the most important sanity checks
across the entire ML workflow. With minor modifications (dataset specificities, different algorithms), 
this tool can be used by an individual, or a small team of data scientists.

2. Most learning courses are too specific:
- There are many courses on how to do regression or classification, or how to write python code
- There are many tutorials on how to use specific algorithms, and how they work under the hood
- There are very few courses/tutorials explaining the ML workflow in a simple manner
- It is very hard to find a platform for quick experimentation to understand how ML workflows work

This project can also serve as a learning tool for understanding ML workflows beyond notebook-based experimentation.
It is easy to set up, and comes with a friendly UI, as well as some pre-saved artifacts for quick inspection.
Users can quickly experiment and learn on their own, and the only assumption is that they know how
to either set up Docker, or python and conda.

## Inspiration

This project started as part of my master's thesis, where the initial goal was to train several models on a hotel booking 
dataset and expose them as tools for an LLM.

While working on that, I quickly ran into practical issues that are common in real-world ML work but rarely addressed in tutorials:
- Repetitive boilerplate for training and evaluation
- Difficulty reusing pipelines across slightly different setups
- Fragile experiment tracking (risk of losing artifacts or overwriting results)
- Inability to reliably pause and resume long-running experiments
- Lack of structure when working beyond notebooks

To address these problems, I started building small utilities to make experimentation more reliable and less error-prone. Over time, 
this evolved into a broader system focused on reproducibility, modularity, and traceability across the entire ML lifecycle.

At some point, it became clear that building a proper ML workflow system was a more meaningful direction than the original project idea, 
so I leaned into it and expanded the architecture into what it is today.

## Key Achievements

- **~17,500** lines of production code
- **~29,000** lines of tests (auto-generated + custom)
- **Fully reproducible pipelines** via artifact hashing
- **End-to-end ML lifecycle support**
- **4,000+** lines of pre-included configurations
- Easy-to-use **ML service** (as a local web app)
- Comprehensive documentation (**3,000+** lines of Markdown)

## Features

### End-to-End ML Pipelines:
- Data registration and preprocessing
- Feature (set) freezing
- Hyperparameter search
- Model training, evaluation and explainability
- Model promotion and archiving
- Model inference and monitoring

### Reproducibility & Validation
- Artifact hashing across pipelines
- Environment & runtime validation
- Heavy versioning:
  - All configurations
    - Interim and processed data configurations
    - Feature registry
    - Global and algorithm defaults
    - Model specifications + search and training configurations
    - Pipeline configurations
    - Environment overlay
    - Promotion thresholds
    - Snapshot bindings
  - Target creation
    - Splitting and target creation performed at runtime, based on model specifications
  - Inference predictions schema
- Heavily snapshot-based:
  - datasets
  - feature sets
  - training, evaluation, and explainability runs
  - promotion and post-promotion runs

### Modular Architecture
- Decoupled datasets, features, and models
- Runtime datasets (DAG + configurations) and feature sets (entity key + configurations) merging
- Flexible snapshot bindings

### Reliability
- Atomic file writing
- Runtime saving of best hyperparameters from each search phase (broad + narrow)
- Runtime saving of model snapshots during training (e.g. every 30 seconds)

### Code Quality
- CI with linting (ruff), typing (mypy), and structure checks
- **90%+** coverage enforced by CI across **1,500+** tests

## Example Use Case

A data scientist can:
1. Register a new dataset snapshot
2. Optimize its memory in one or more ways
3. Process the dataset in one or more ways
4. Define and freeze many feature sets, each based on one or more related datasets
5. Perform one or more hyperparameter searches
6. Train models based on the hyperparameter search results (many training runs allowed per each search)
7. Evaluate and explain the trained models, however many times
8. Stage, promote, and archive models
9. Run inference and monitoring on incoming data

## Installation

See the [setup guide](docs/setup.md) for installation instructions.

### Brief version

Two options:
- **Docker**
  - requires `docker`
  - operate the workflow through a `browser`
- **Manual use**
  - requires `python` and `conda` (preferably)
  - operate the workflow through a `browser` or `cli + manual (configs)`

## Usage

See the [usage guide](docs/usage.md) for instructions on running the workflow.

### Usage examples:

The system includes a browser-based interface (`ml_service`) for interacting with pipelines and configurations:

#### Configs Writing, Validation, Saving, and Viewing - Interim Data Configs Example

!["Gif portrayal of writing interim data configs with ml_service"](assets/gifs/ml_service_data_configs_v1.gif)

**Similar functionality exists for other supported configs**

#### Pipeline Running and Artifact Viewing - freeze.py Example

!["Gif portrayal of running a pipeline with ml_service"](assets/gifs/ml_service_pipelines_v3.gif)

**Similar functionality exists for scripts**

#### Documentation Reading in Browser

!["Gif portrayal of reading the docs with ml_service"](assets/gifs/ml_service_docs_v1.gif)

## Architecture

### Artifact Lineage (high-level overview)

![Artifact Lineage Diagram](assets/img/docs/architecture/artifact_lineage_v3.png)

### Details

See the [architecture overview](docs/architecture/overview.md) for details, including:
- Artifact lineage of each pipeline
- Architectural decisions and reasoning
- Validation guarantees
- System invariants
- Boundaries

## Documentation

Full documentation is in [`docs/`](docs/README.md).
It includes:
- Architectural details (mentioned earlier)
- Configuration details (what each config expects)
- Glossary
- Maintenance guidance
- Roadmap
- Setup instructions
- Testing details
- Usage instructions
- API docs (generated with the `pdoc` package)

## Contributing

Please read [`CONTRIBUTING.md`](.github/CONTRIBUTING.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Sebastijan Dominis

## Contact

sebastijan.dominis99@gmail.com