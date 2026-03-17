# Hotel Management

## Overview

***A reproducible ML experimentation and model lifecycle system.***

- Currently supports the modeling of regression and classification tasks using the CatBoost algorithm.
- Was initially formed based on a hotel_bookings dataset:
    - located in `data/raw/hotel_bookings/v1/2026-02-25T22-43-23_732dfdb7/data.csv`
    - originally from https://www.kaggle.com/datasets/mojtaba142/hotel-booking 
- Current architecture expanded to support many datasets.
- The ml workflow covers everything from the registration of a raw data snapshot to model promotion.
> Note: the repo was previously named `hotel_management`, so you will see that name around the repo; renamed for clarity on what the project does

## Features

- Data preprocessing
  - Register raw data snapshots
  - Build interim and processed datasets
- Feature (set) freezing
- Hyperparameter search
- Model training
- Model evaluation
- Model explainability
- Model promotion
  - Includes model registry for staging and production
  - Archives past production models

## Installation

See the [setup guide](docs/setup.md) for installation instructions.

### Brief version

Two options:
- **Docker**
  - requires `docker`
  - operate the workflow through a `browser`
- **Manual use**
  - requires `python` and `conda` (preferably)
  - operate the workflow through a `browser` or `cli + manual` (configs)

## Usage

See the [usage guide](docs/usage.md) for instructions on running the workflow.

### Usage examples (via `ml_service` and `Docker`):

#### Pipelines

!["Gif portrayal of pipelines app from ml_service"](assets/gifs/ml_service_pipelines_v2.gif)

#### Modeling Configs

!["Gif portrayal of modeling configs app from ml_service"](assets/gifs/ml_service_modeling_configs_v2.gif)

## Architecture

### Artifact Lineage (high-level overview)

![Artifact Lineage Diagram](docs/architecture/img/artifact_lineage_v2.png)

### Config Resolution

![Config Resolution Diagram](docs/architecture/img/config_resolution_v3.png)

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