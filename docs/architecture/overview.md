# Architecture Overview

This document provides a high-level overview of the system architecture, including artifact lineage, orchestration flow, config resolution, and each of the pipeline scripts (from registering raw data to promotion).

## Main Components
- Pipelines: Orchestration and CLI logic
- ML: Business/domain logic
- Configs: Declarative YAML configuration
- Scripts: Useful scripts (quality assessment, hash generation, etc.)

## Diagrams

### Artifact Lineage (high-level overview)

![Artifact Lineage Diagram](img/artifact_lineage.png)

### Orchestration (high-level overview)

![Orchestration Architecture Diagram](img/orchestration.png)

### Config Resolution

- Applicable for search and train

![Config Resolution Architecture Diagram](img/config_resolution.png)

### Data Preprocessing

#### Raw Snapshot Registration

![register_raw_snapshot.py Architecture Diagram](img/register_raw_snapshot.png)

#### Interim Dataset Building

![build_interim_dataset.py Architecture Diagram](img/build_interim_dataset.png)

#### Processed Dataset Building

![build_processed_dataset.py Architecture Diagram](img/build_processed_dataset.png)

### Feature Freezing

#### Feature Set Freezing

![freeze.py Architecture Diagram](img/freeze.png)

### Search

#### Hyperparameter Search

![search.py Architecture Diagram](img/search.png)

### Runners

#### Training

![train.py Architecture Diagram](img/train.png)

#### Evaluation

![evaluate.py Architecture Diagram](img/evaluate.png)

#### Explainability

![explain.py Architecture Diagram](img/explain.png)

### Promotion

#### Model Staging, Promotion and Archiving

![promote.py Architecture Diagram](img/promote.png)

## Design Decisions
- See [decisions.md](decisions.md) for rationale behind architectural choices.
