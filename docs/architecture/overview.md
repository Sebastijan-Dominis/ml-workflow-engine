# Architecture Overview

This document provides a high-level overview of the system architecture, including artifact lineage, orchestration flow, config resolution, and each of the pipeline scripts (from registering raw data to promotion).

## Main Components
- Pipelines: Orchestration and CLI logic
- ML: Business/domain logic
- Configs: Declarative YAML configuration
- Scripts: Useful cli scripts that don't belong to ml or pipelines (quality assessment, hash generation, etc.)

## Diagrams

### Artifact Lineage (high-level overview)

![Artifact Lineage Diagram](/assets/img/docs/architecture/artifact_lineage_v3.png)

### Orchestration (high-level overview)

![Orchestration Architecture Diagram](/assets/img/docs/architecture/orchestration_v2.png)

### Config Resolution

- Applied in search.py and train.py

![Config Resolution Architecture Diagram](/assets/img/docs/architecture/config_resolution_v3.png)

### Data Preprocessing

#### Raw Snapshot Registration

![register_raw_snapshot.py Architecture Diagram](/assets/img/docs/architecture/register_raw_snapshot_v2.png)

#### Interim Dataset Building

![build_interim_dataset.py Architecture Diagram](/assets/img/docs/architecture/build_interim_dataset_v2.png)

#### Processed Dataset Building

![build_processed_dataset.py Architecture Diagram](/assets/img/docs/architecture/build_processed_dataset_v2.png)

### Feature Freezing

#### Feature Set Freezing

![freeze.py Architecture Diagram](/assets/img/docs/architecture/freeze_v3.png)

### Search

#### Hyperparameter Search

![search.py Architecture Diagram](/assets/img/docs/architecture/search_v3.png)

### Runners

#### Training

![train.py Architecture Diagram](/assets/img/docs/architecture/train_v3.png)

#### Evaluation

![evaluate.py Architecture Diagram](/assets/img/docs/architecture/evaluate_v2.png)

#### Explainability

![explain.py Architecture Diagram](/assets/img/docs/architecture/explain_v2.png)

### Promotion

#### Model Staging, Promotion and Archiving

![promote.py Architecture Diagram](/assets/img/docs/architecture/promote_v2.png)

### Post-promotion

#### Inference

![infer.py Architecture Diagram](/assets/img/docs/architecture/infer_v1.png)

#### Monitoring

![monitor.py Architecture Diagram](/assets/img/docs/architecture/monitor_v1.png)

## Design Decisions
- See [decisions.md](decisions.md) for rationale behind architectural choices.

## Boundaries
- See [boundaries.md](boundaries.md) for boundaries-specific information.

## System Invariants
- See [system_invariants.md](system_invariants.md) for details on system invariants.

## Validation Guarantees
- See [validation_guarantees.md](validation_guarantees.md) for details on validation guarantees.