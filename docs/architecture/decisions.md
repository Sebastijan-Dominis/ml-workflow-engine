# Architectural Decisions

This file records key architectural decisions, their rationale, and alternatives considered.

## Key Architectural Decisions (Summary)

The system is built around a few core principles:

- **Immutability of artifacts** (datasets, features, experiments)
- **Full reproducibility via configs + snapshot IDs**
- **Decoupling of datasets, features, and models**
- **Snapshot-based versioning instead of mutable state**
- **Filesystem-based storage with strict validation**

These decisions shape the entire architecture. Detailed breakdowns are provided below.

## Decision Classification

Each decision is classified as one of:

- **Foundational** – Changing this would require major architectural refactoring.
- **Structural** – Impacts multiple subsystems but can be migrated with effort.
- **Tactical** – Can be changed with limited refactoring.
- **Convenience** – Organizational preference; low migration cost.

## Storage assumption
The system assumes:
- Single-writer per experiment
- Atomic file writes for artifacts

Migration to object storage or distributed runners would require
a storage abstraction layer.

## System Invariants

- Experiments are immutable once completed.
- Snapshots are immutable.
- All artifacts are reproducible from configs + snapshot ids.
- No pipeline mutates upstream artifacts.

## Design Philosophy

- Product-oriented
- Reproducibility over convenience
- Explicit versioning over implicit mutation
- Canonical lineage between modeling stages
- Filesystem transparency over hidden metadata services

## Snapshot Policy (Applies to raw/interim/processed/features)
Snapshots represent new rows for the same logical dataset(s) version(s).
Versions represent schema or transformation changes.
Snapshots are immutable.

## General/global

### Separation of `ml` from `pipelines`
- **Decision:** Keep ML logic independent from pipelines
- **Rationale:** Promotes reusability and testability
- **Alternatives:** Integrate ML logic into pipelines (rejected - less reusable)
- **Type:** Structural (would imply changing pre-commit quality checks, as well as how the code within notebooks/ is used)

### Logs stored at the lowest level
- **Decision:** Store logs in the lowest-level folder that was created by a given run of a pipeline
- **Rationale:** Easy to understand what the logs refer to, easy to navigate the logs
- **Alternatives:** Create a logs folder at the repo root level, then branch from there (rejected - mostly harmless, but unnecessary and harder to navigate and maintain); log to 
                    the same location depending on the pipeline (e.g. all calls to `freeze.py` log to `freeze.log`) (rejected - quickly leads to logging hell and extremely hard 
                    to navigate)
- **Type:** Convenience (could just change the lines that define the log level)

### Defining the logging level through a cli argument
- **Decision:** Define the logging level (DEBUG/INFO...) through a cli argument for each given pipeline
- **Rationale:** Flexible, with no clear downsides (e.g. use DEBUG for development and INFO for production)
- **Alternatives:** Always use the same logging level (rejected - very inflexible); Define the logging level in configs (rejected - unnecessarily complicated; logging should not 
                    depend on configs, and should be more flexible)
- **Type:** Convenience (defining a default is easy)

### Decoupling of feature sets and data
- **Decision:** Keep feature sets and data decoupled, even though features depend on processed datasets
- **Rationale:** Allows for feature sets to be composed of features from various datasets
- **Alternatives:** Always use the same dataset for the same feature set (rejected - feature in different datasets may be logically similar; less flexible)
- **Type:** Foundational

### Decoupling of modeling and feature freezing
- **Decision:** Decouple feature freezing from modeling (search, runners)
- **Rationale:** Makes the features model-agnostic and reusable, allows for different models and their versions to use different feature sets; very scalable
- **Alternatives:** Tie features to each model (rejected - less scalability, flexibility and reusability)
- **Type:** Foundational

### Snapshot id enabled as a cli argument in data preprocessing pipelines
- **Decision:** Include snapshot id (representing data snapshot from the previous stage) as an optional cli argument; default to latest
- **Rationale:** Enables flexibility and scalability with little overhead
- **Alternatives:** Always use the latest snapshot (rejected - no reason to force this)
- **Type:** Structural

### Flexible snapshot usage for each of the feature set's datasets (via snapshot bindings)
- **Decision:** Use snapshot bindings cli argument to enable the user to freeze a feature set using different combinations of dataset snapshots
- **Rationale:** Flexible, scalable, reproducible
- **Alternatives:** Force the use of latest snapshots (rejected - not flexible and scalable enough; prevents proper reproducibility)
- **Type**: Structural

### Flexible snapshot usage for each of the experiment's feature sets
- **Decision:** Allow each experiment to use dataset and feature set snapshots defined by the user via a cli argument
- **Rationale:** Flexible, scalable, reproducible
- **Alternatives:** Always use the latest full feature set snapshots when starting a new experiment (rejected - same reason as in [feature freezing](#flexible-snapshot-usage-for-each-of-the-feature-sets-datasets-via-snapshot-bindings))
- **Type:** Structural

### Flexible snapshot usage for training runs
- **Decision:** Allow each training run to use dataset and feature set snapshots defined by the user via a cli argument
- **Rationale:** Flexible, scalable, reproducible
- **Alternatives:** Same as with the [experiments](#flexible-snapshot-usage-for-each-of-the-experiments-feature-sets)
- **Type:** Structural

### Absence of the inference and monitoring pipelines
- **Decision:** Do not implement inference and monitoring pipelines
- **Rationale:** The current snapshot logic (full snaphots with old and new rows) renders these two obsolete; not needed at this stage
- **Alternatives:** Run inference and monitoring on all rows (those used for training the model + new ones) (rejected - results could be misleading)
- **Type:** Structural

### Absence of cron jobs
- **Decision:** Do not implement cron jobs
- **Rationale:** The current architecture implies they are not needed
- **Alternatives:** Run `run_all_workflows.py` whenever a new snapshot of raw data appears (rejected - not needed at this stage; could result in unexpected 
                    outcomes - that orchestrator is powerful and potentially dangerous)
- **Type:** Tactical

## Artifacts and Configs - General

### Informational artifacts stored in `json` format
- **Decision:** Store all of the informational artifacts (metadata, runtime info, metrics) in `json` format
- **Rationale:** Easy to validate, read and work with
- **Alternatives:** Use yaml instead (rejected - json is more readable and often considered standard)
- **Type:** Tactical (switching to a different format would only impact saving and loading)

### Configs stored in `yaml` format
- **Decision:** Store all of the configs in `yaml` format
- **Rationale:** Easy to validate, read and work with
- **Alternatives:** No alternatives were considered for this decision
- **Type:** Tactical

### Configs stored in the `configs/` folder
- **Decision:** Store all of the configs within the `configs/` folder, with relevant, semantic nesting
- **Rationale:** Easy to access, work with, and reason about
- **Alternatives:** Use sql instead (rejected - a viable alternative at great scale, but would add unnecessary complexity to the current system; 
                    migrating to it later on is simple enough)
- **Type:** Structural (pipelines and orchestrators assume the current structure when searching for configs)

### Experiments and all of their artifacts stored in the `experiments/` folder
- **Decision:** Store all of the experiments and the artifacts they produce (e.g. metadata, runtime info, metrics, models, pipelines) within the 
                `experiments/` folder
- **Rationale:** Easy to access, work with, and reason about
- **Alternatives:** Store models and pipelines separately (rejected - models and pipelines are strongly tied to training runs, and decoupling them would 
                    decrease safety and add complexity); use cloud services (rejected - great option long-term, but would incur unnecessary costs at the 
                    current stage)
- **Type:** Structural (pipelines assume the current structure when loading artifacts)

### Metadata produced by each pipeline (e.g. `freeze.py`)
- **Decision:** Every pipeline produces some metadata
- **Rationale:** Forces good practices, like auditability
- **Alternatives:** Only produce metadata when expected to be used by downstream code (rejected - considered to be a bad practice)
- **Type:** Tactical (removing the creation of metadata that is not used downstream would simply imply the deletion of some code)

### Storing of runtime information for feature freezing and experiments
- **Decision:** Store runtime information for each run of `freeze.py`, `search.py`, `train.py`, `evaluate.py` and `explain.py` (`freeze.py` includes it in metadata, 
                while the other four produce a separate artifact)
- **Rationale:** Usually cheap to generate and store; improves auditability and reproducibility
- **Alternatives:** Do not store runtime information (rejected - the benefits outweight the costs)
- **Type:** Structural (used for reproducibility validations in multiple pipelines)

## Modeling

### Models defined by problem type, segment and version
- **Decision:** Define the models by the problem they solve, the segment they encompass, and the version of those two (based on configs)
- **Rationale:** Allows for seamless navigation and scalability, very future-proof
- **Alternatives:** Define only by problem type or problem type and segment (rejected - less scalable, harder to navigate)
- **Type:** Foundational (all of modeling and promotion logic assumes the current structure)

### Nesting search and training configs same as model specs
- **Decision:** Use the same folder nesting for search and training configs as with model specs configs
- **Rationale:** Easy to navigate, seamless config resolution at runtime
- **Alternatives:** Version search/train configs separately (rejected - unnecessary complexity)
- **Type:** Structural (could move everything into model specs f.x., but that would require changing the config resolution logic; versioning search and train separately 
            would imply major restructuring)

### Search/training final configs overriding order: global defaults -> algorithm defaults -> model specs -> search/train -> env (optional) -> best params from search (only used by `train.py`)
- **Decision:** When forming the final configs for search or training, first override the global defaults by algorithm defaults, then by model specs, then by search or training configs, and 
                finally by environment configs (optional); overridden again by best params from search during training runs
- **Rationale:** Defaults are always good to have for future-proofing, model specs define most of the behavior, search and training configs define information specific to their stage of 
                modeling, environment configs allow easy override of the most impactful elements (e.g. number of iterations); more reusable and less error prone when search and training 
                use the same logic
- **Alternatives:** Do not include defaults and/or environment configs (rejected - less scalable; those configs are extremely inexpensive to store, but can easily come in handy); decouple 
                    search from training config creation (rejected - more error prone, more boilerplate code)
- **Type:** Structural (could add or remove a step, but should be carefully thought through - impacts the config resolution that ultimately determines search and training runs)

### Passing of env as a cli argument
- **Decision:** Pass environment as a cli argument in relevant pipelines
- **Rationale:** Dev and test allow for quickly iterating and testing the pipelines, while prod ensures the required quality in production - more flexible to use through cli
- **Alternatives:** Do not use env at all (rejected - env override is extremely useful)
- **Type:** Structural (technically easy to change, but likely implies that future dev/test/prod separation means one extra model for each, which adds complexity)

### Feature engineering decided at feature freezing
- **Decision:** Decide which engineered features to use for a given version of a given feature set through the feature registry
- **Rationale:** Ensures that the input features necessary to create the engineered/derives features are always present by raising an error early (during feature freezing) if they are not
- **Alternatives:** Engineer features during data preprocessing (rejected - could lead to a bad downstream situation where a model's pipeline wants to create a feature, but does not have 
                    the necessary input - e.g. it wants to create adr_per_person, but adr is not present)
- **Type:** Structural (pipeline (sklearn) creation fully relies on this assumption)

### Operators defined by derived schemas
- **Decision:** Create operators (==engineer features) at runtime from derived schemas
- **Rationale:** Ensures that the input features necessary to create the engineered features are always present, given that the derived schemas are created programmatically by `freeze.py`
- **Alternatives:** Manually define operators via configs (rejected - look at [Feature engineering decided at feature freezing](#feature-engineering-decided-at-feature-freezing))
- **Type:** Structural

### Target creation at runtime
- **Decision:** Create the target variable at runtime
- **Rationale:** Enables seamless versioning of target creation; simplifies validation
- **Alternatives:** Freeze the target variable during the freezing step (rejected - target creation may require features from different datasets; makes versioning and validation more challenging)
- **Type:** Structural

### Target versioning
- **Decision:** Version how the target variable is created
- **Rationale:** Although rarely, we may sometimes want to create the target for a new version of the same problem type and segment in a different way
- **Alternatives:** Always create the target for the same model in the same way (rejected - less flexible and future-proof)
- **Type:** Structural

## Orchestration

### Running all data preprocessing (raw/interim/processed) with a single orchestrator enabled
- **Decision:** Enable the execution of all data preprocessing for all of the available datasets and configs (the orchestrator figures it out)
- **Rationale:** Drammatically speeds up the data preprocessing step when needed
- **Alternatives:** Only ever use individual pipelines (e.g. build_interim_dataset.py) (rejected - can be time-inefficient; proper validations are in check anyway)
- **Type:** Convenience (can easily delete the orchestrator and related tests)

### Freezing all feature sets from the feature registry with a single orchestrator enabled
- **Decision:** Enable the freezing of all feature sets found in the feature registry by running a single orchestrator
- **Rationale:** More efficient when needed
- **Alternatives:** Force freezing feature sets individually (rejected - can be time-inefficient; proper validations are in check anyway)
- **Type:** Convenience

### Executing the entire experiment for a single model with a single orchestrator enabled
- **Decision:** Enable a sequential execution of search, training, evaluation and explainability for a given model by defaulting to the latest snapshots for training, evaluation and explainability steps
- **Rationale:** More efficient when needed
- **Alternatives:** Force running the pipelines individually (rejected - the runs log which experiment id and training id they defaulted to anyway, so tracking it is still straightforward)
- **Type:** Convenience

### Executing all workflows with a single orchestrator enabled
- **Decision:** Enable running all data preprocessing for all of the available datasets and configs, followed by freezing of all feature sets found in the feature registry, followed by the execution of 
                search, training, evaluation and explainability for all of the available models (orchestrator figures it out by inspecting how `configs/model_specs/` is nested) by defaulting to latest 
                snapshots for training, evaluation and explainability steps
- **Rationale:** Extremely efficient when appropriate
- **Alternatives:** Forbid such high-level orchestration (rejected - the user is responsible (and warned) for what happens next; proper validations are in check and the logs enable proper tracing)
- **Type:** Convenience

### Orchestrating promotion disabled
- **Decision:** Do not orchestrate the promotion step
- **Rationale:** Hard to implement if meant to be run with the grand orchestrator; easy to result in unexpected outcomes; usually not needed
- **Alternatives:** Create `stage_all.py` or `promote_all.py` and run it right after `execute_all_experiments_with_latest.py` (rejected - hard to implement, as we don't know which 
                    experiment/training/evaluation/explainability ids to use prior to running the grand orchestrator; rarely needed; could end up polluting the model registry, which 
                    is a terrible idea, as that is the source of truth for staging and production models)
- **Type:** Foundational (very important to keep the model registry clean)

### Orchestrating post-promotion pipelines disabled
- **Decision:** Do not orchestrate the post-promotion pipelines (inference and monitoring)
- **Rationale:** Those two steps should be intentional, implicitly prompting the user to inspect the produced `json` artifacts at each run
- **Alternatives:** Orchestrate the two steps by defaulting to latest, similar to the experiment orchestrator (rejected - a viable idea,
                    but requiring a few extra steps from the user at these steps is actually desirable, as it reduces the likelihood of them
                    running the pipelines without properly inspecting the output)
- **Type:** Tactical

## Data

### Raw data versioning
- **Decision:** Use versioning for raw data
- **Rationale:** Adding or removing columns within the same dataset simply implies a new version - logically it is the same dataset
- **Alternatives:** Imply a new version in the name (e.g. hotel_bookings_v2) (rejected - semantically complicated)
- **Type:** Structural

### Snapshots for raw data
- reference [Snapshot policy](#snapshot-policy-applies-to-rawinterimprocessedfeatures)

### Separation of raw and interim data
- **Decision:** Keep interim data separately from raw
- **Rationale:** Raw data is the source of truth. Interim optimizes for efficiency, while preserving the source of truth.
- **Alternatives:** Optimize and overwrite raw data immediately (rejected - loses the source of truth)
- **Type:** Foundational (very important to have a reliable source of truth)

### Interim datasets versioning
- **Decision:** Use versioning for interim datasets
- **Rationale:** Optimizing raw data by opting for different data types simply implies a new version - logically tied to the same version of the same raw data
- **Alternatives:** Always use the same optimization for each version of each raw data (rejected - less flexible with no clear upsides); Imply new version in the name 
                    (e.g. hotel_bookings_v2) (rejected - semantically complicated)
- **Type:** Structural

### Snapshots for interim datasets
- reference [Snapshot policy](#snapshot-policy-applies-to-rawinterimprocessedfeatures)

### Separation of interim and processed data
- **Decision:** Keep processed data separately from interim
- **Rationale:** Interim is used for memory optimization, while processed takes care of the necessary transformations.
- **Alternatives:** Optimize and process in the same step (rejected - we often want to process differently, but optimize in the same way)
- **Type:** Structural (not necessarily foundational, but very hard to properly implement otherwise)

### Processed datasets versioning
- **Decision:** Use versioning for processed datasets
- **Rationale:** Applying different preprocessing simply implies a new version - logically tied to the same version of the same interim dataset
- **Alternatives:** Always preprocess a given version of an interim dataset in the same way (rejected - extremely unflexible and impractical); Imply new version 
                    in the name (e.g. hotel_bookings_v2) (rejected - semantically complicated)
- **Type:** Structural

### Snapshots for processed datasets
- reference [Snapshot policy](#snapshot-policy-applies-to-rawinterimprocessedfeatures)

## Features

### Division into feature sets
- **Decision:** Create feature sets by combining only the logically similar features
- **Rationale:** Promotes explainability, reusability, and scalability
- **Alternatives:** Use generic "features" with many different versions (rejected - quickly leads to versioning hell, hard to interpret)
- **Type:** Foundational (all of feature freezing and modeling assumes this; change would radically deteriorate the entire system)

### Feature set versioning
- **Decision:** Version feature sets
- **Rationale:** Quick and easy adaptation to new modeling needs
- **Alternatives:** Always use the same set of features in a feature sets (rejected - extremely bad practice, since different models often imply different feature needs)
- **Type:** Foundational (feature registry - the source of truth for all of feature freezing - is based on this assumption)

### Snapshots for feature sets
- reference [Snapshot policy](#snapshot-policy-applies-to-rawinterimprocessedfeatures)

### Merging of datasets on multiple keys, with flexible logic, using DAG
- **Decision:** Allow different kinds of joins, along with multiple keys, for dataset merging
- **Rationale:** Keeps the complexity at this step, rather than propagating it down to feature sets; allows for all of the required flexibility
                with minimum overhead
- **Alternatives:** Force a specific type of merging with only one key (rejected - while currently viable, this solution scales poorly, and
                    makes no sense given that the implementation of the more flexible merging logic is not that much more complex); make
                    the feature sets handle the complexity (rejected - feature set merging is more challenging; it is better to keep the
                    merging complexity upstream)
- **Type:** Structural

### Keeping of the entity key in frozen features
- **Decision:** Keep entity key column when freezing any given feature set
- **Rationale:** Allows for seamless merging of many feature sets at runtime, regardless of their respective row counts
- **Alternatives:** Drop entity key while freezing a feature set (rejected - makes proper merging at runtime impossible)
- **Type:** Foundational (impossible to implement the modeling pipelines without it)

### Saving of input and derived schemas at the version level
- **Decision:** Save both input and derived schemas once at the version level (one for all snapshots)
- **Rationale:** Schemas depend on configs, which should not change across runs; more logical and saves memory
- **Alternatives:** Save new schemas for each snapshot (rejected - redundant, could be more error prone if the code did not include all of the validations that it already does)
- **Type:** Structural (changing the validation logic is simple enough, but changing sklearn pipeline building logic would break all of the existing pipelines that assumed the 
            existing architecture; technically possible to keep old logic for older ones, and create new for the newer ones - but a lot of work)

### The disinclusion of entity key in input_schema
- **Decision:** Do not include entity key in feature set's input schema
- **Rationale:** Entity key is only used for merging, should never reach the model, and does not provide any value outside of merging; this decision aligns well with the 
                rest of the code logic
- **Alternatives:** Include entity key in input schema (rejected - it is only a technical feature)
- **Type:** Structural (implies changes to sklearn pipeline building logic - would impact older pipelines)

### Forcing of a single entity key per feature set
- **Decision:** Force the use of one single entity key per feature set
- **Rationale:** Entity key inclusion already provides enough flexibility, while this decision provides seamless merging upstream without compromising
                the said flexibility
- **Alternatives:** Allow multiple entity keys, as in a list (rejected - completely unnecessary at this stage, and would require major refactoring)
- **Type:** Structural

### Not saving derived schema when no operators in configs
- **Decision:** Do not save a derived schema if no operators are being created, meaning all of the features are already included in the input schema
- **Rationale:** Avoids flooding the feature store with empty derived schemas
- **Alternatives:** Save an empty derived schema instead (rejected - harmless, but unnecessary)
- **Type:** Tactical

### Hardcoded coverage threshold when merging feature sets
- **Decision:** Hardcode a coverage threshold in feature set merging
- **Rationale:** Not meant to be changed; should be a good default for all datasets and feature sets; doesn't prevent code execution anyway
- **Alternatives:** Include coverage threshold in model specs (rejected - not needed; one global default works just fine)
- **Type:** Tactical

### Missing the coverage threshold only logs a warning
- **Decision:** Instead of blocking the execution in case of a missed coverage threshold, just log a warning and continue
- **Rationale:** Low coverage threshold does not automatically mean that there is something wrong, but the user should
                be notified in some way anyway - they can then inspect what is happening on their own if needed
- **Alternatives:** Fail loudly if the coverage threshold is not met (rejected - too rigid); do not include the threshold
                    at all (rejected - informing the user is useful and easy to implement)
- **Type:** Tactical

## Snapshot bindings

### Implementation of snapshot bindings
- **Decision:** Implement snapshot bindings to allow merging of specific feature sets and datasets, rather than defaulting to latest on every run
- **Rationale:** Considerably more flexible and scalable; allows for proper implementation of upstream code (runners, inference, monitoring)
- **Alternatives:** Always default to latest snapshots (rejected - very limiting and not scalable); include snapshot definitions in the
                    feature registry and model specs (rejected - quickly leads to config hell, since it requires constant adding of new configs)
- **Type:** Foundational

### Inclusion of snapshot bindings generation script
- **Decision:** Include a script for generating snapshot bindings, while defaulting to latest
- **Rationale:** Allows for an easy creation of new snapshot bindings that does not require too much manual work; manual work may still be required,
                but is generally greatly reduced with the inclusion of this script
- **Alternatives:** Force manual writing of the snapshot bindings (rejected - error-prone and difficult)
- **Type:** Convenience

### Grouping of many datasets and feature sets in snapshot bindings
- **Decision:** Group as many datasets and feature sets as desired in each snapshot binding
- **Rationale:** It doesn't matter if a pipeline uses only some of them - extras don't hurt; including more in one reduces the chances of
                config explosion, since it implies less snapshot bindings are needed
- **Alternatives:** Separate dataset and feature set snapshot bindings (rejected - this would actually be more error-prone, since models need both;
                    including both makes the implementation safer and easier)
- **Type:** Tactical

## Search

### One broad and one optional narrow search
- **Decision:** Always do one broad hyperparameter search, optionally followed by one narrow search
- **Rationale:** Flexible enough to only do one search for models that do not require more than that; there is no need for more than two searches - increasing iterations 
                is a better approach
- **Alternatives:** Allow the configs to define the number of searches (rejected - unnecessary complexity; just increase iterations or improve data if results are subpar); 
                    always do both broad and narrow search (rejected - some models converge very quickly and doing two searches is a waste of time and resources in that case)
- **Type:** Structural (implies major changes to all of the search-related code)

### Forced use of one search type per algorithm
- **Decision:** Always use the same search type for each algorithm (e.g. randomized search for Catboost)
- **Rationale:** Enhances reproducibility; easier to understand the results; minimal harm; still flexible on algorithm-level 
- **Alternatives:** Use one search type for all algorithms (rejected - not flexible enough); allow configs to define search type for each model (rejected - the ROI is not 
                    there - implementation and interpretation become more difficult, while the gains are minimal and irrelevant for a product-oriented system)
**Type:** Structural (implies major changes to all of the search-related code)

### Parameter distribution for broad search defined in configs
- **Decision:** Define the parameter distributions in search configs
- **Rationale:** Provides flexibility, especially when we have an idea of where a given hyperparameter could end up; scalable
- **Alternatives:** Define parameter distributions in code, and use the same ones with all models (rejected - less efficient, less flexible)
- **Type:** Tactical (code-based implementation is the easier solution; no downstream code is impacted by this)

### Parameter configurations for narrow search defined in configs
- **Decision:** Define the configurations that take the best hyperparameters from broad search at runtime and create parameter distrubutions for narrow search in search configs
- **Rationale:** Same as in [Parameter distribution for broad search defined in configs](#parameter-distribution-for-broad-search-defined-in-configs)
- **Alternatives:** Always use the same numbers to calculate parameter distributions for the narrow search (rejected - less efficient, less flexible)
- **Type:** Tactical (downstream code is decoupled from this)

### Saving of best hyperparameters from broad and narrow search respectively at runtime
- **Decision:** Save the best hyperparameters upon finishing both broad and narrow search respectively, at runtime; saved in the failure management folder
- **Rationale:** Allows to seamlessly continue with either narrow search or persistence in case the code fails at runtime, or the hardware running it suddenly stops working
- **Alternatives:** Do not save anything when running the search pipeline (rejected - search is expensive, failures with no backup can cost many dollars and hours)
- **Type:** Tactical (most of the code is completely decoupled from this)

### Allowing deletion of the saved best hyperparameters from broad and narrow search
- **Decision:** Enable the saved best hyperparamters from broad and narrow search to be dynamically deleted if the search run finishes successfully, but do not force the deletion
- **Rationale:** Prevents the hoarding of useless information without requiring manual deletion; flexible to allow override (skipping deletion) when desired
- **Alternatives:** Do not delete the saved best hyperparameters from broad and narrow search dynamically (rejected - can easily fill space with useless data, and requires 
                    unnecessary manual intervention); always delete (rejected - less flexible)
- **Type:** Tactical (extremely easy to delete one function, but implies slightly more manual monitoring of storage space)

### Search runs create experiments
- **Decision:** Each search run creates a new experiment
- **Rationale:** Experiments include various artifacts from search and runners, all of which are ultimately defined by the search; hence new search = new experiment
- **Alternatives:** Enable multiple search runs per experiment (rejected - this is often done in research-focused ml-infra, since running the same code with the same configs on 
                    slightly different hardware may result in slightly different floating points - this is a product-oriented system, so those tiny differences are considered to be irrelevant)
- **Type:** Foundational (all of modeling assumes this)

### Search run artifacts stored in `experiments/.../{experiment_id}/search/`
- **Decision:** Store all of the artifacts generated by a search run in `experiments/{problem_name}/{segment}/{model_version}/{experiment_id}/search/`
- **Rationale:** Search is canonical for a given experiment, so there is no need for snapshots; nevertheless, including it within a nested search folder makes it semantically clear that the 
                artifacts were generated by the search
- **Alternatives:** Create a nested search run for storing artifacts created by a search run (rejected - reasoning explained already); Store the generated artifacts at the experiment folder 
                    level (rejected - viable alternatives, but less semantically accurate)
- **Type:** Tactical (implies changing a few paths downstream, likely an easy implementation)

## Training

### Experiment (created by a search run) canonical for training runs
- **Decision:** Tie the training run to a given experiment (generated by a search run)
- **Rationale:** Training needs the best parameters found from search to assemble the final runtime configs; results will greatly differ based on that
- **Alternatives:** Decouple training from search (rejected - training without using the best params from any given search run is useless)
- **Type:** Foundational (training should never be done without a prior search)

### Snapshots for training
- **Decision:** One experiment can have many training runs; each stores its artifacts in `experiments/{problem_name}/{segment}/{model_version}/{experiment_id}/training/{train_run_id}/`
- **Rationale:** Training with the same configs and hardware can result in a different model if we train on updated data; this prevents wasting resources on re-searching params when that 
                is not necessary
- **Alternatives:** One training run per experiment (rejected - requiring each training run to imply a new search run is extremely inefficient)
- **Type:** Foundational (necessary to have an efficient system)

### Saving model snapshots as backup while training
- **Decision:** Save a model snapshot every x seconds (defined by configs) while the training is being executed
- **Rationale:** In case of a system failure, we can continue training from where we left off, rather than start from the start (e.g. we can continue from iteration 987, rather than 0)
- **Alternatives:** Not saving anything during training (rejected - training can be expensive, and relying on the belief that the execution will surely go well every time is dangerous)
- **Type:** Tactical (implies changing a few lines of code, the rest of it is decoupled)

### Allowing deletion of the training backups
- **Decision:** Enable the training backups (model snapshots) to be dynamically deleted if the training run finishes successfully, but do not force the deletion
- **Rationale:** Same as [Allowing deletion of the saved best hyperparameters from broad and narrow search](#allowing-deletion-of-the-saved-best-hyperparameters-from-broad-and-narrow-search)
- **Alternatives:** Same as [Allowing deletion of the saved best hyperparameters from broad and narrow search](#allowing-deletion-of-the-saved-best-hyperparameters-from-broad-and-narrow-search)
- **Type:** Tactical (same logic as in [Allowing deletion of the saved best hyperparameters from broad and narrow search](#allowing-deletion-of-the-saved-best-hyperparameters-from-broad-and-narrow-search))

## Evaluation

### Train run canonical for evaluation runs
- **Decision:** Tie the evaluation run to a given train run
- **Rationale:** Evaluation run evaluates a model, so it has to be bound to a training run
- **Alternatives:** No alternatives (we need a model/pipeline as input to have something to evaluate)
- **Type:** Foundational (no viable alternatives if evaluation is done separately from training - which is the intended architecture)

### Snapshots for evaluation
- **Decision:** One trained model can be evaluated many times; each evaluation run stores its artifacts in `experiments/{problem_name}/{segment}/{model_version}/{experiment_id}/evaluation/{eval_run_id}/`
- **Rationale:** While usually the same, evaluation runs can sometimes result in different output (f.x. if we change the code - in that case the warning is logged, but the execution is allowed to continue); this approach prevents the expensive re-training, and allows us to seamlessly re-evaluate our older models with new metrics
- **Alternatives:** Allow only one evaluation run per trained model/pipeline (rejected - dangerous, since adding new metrics in the future may result in hundreds of models requiring another training run)
- **Type:** Structural (implies changes in promotion logic, as well as bad architecture, but technically possible)

### Storing row-by-row predictions by split (train/val/test)
- **Decision:** Store the exact row-by-row predictions the model made for each split (train/val/test)
- **Rationale:** In a case of weird outcomes, this provides an easy way of inspecting where the model is failing; could potentially be re-used in explainability runs (currently not implemented)
- **Alternatives:** Save storage space by not saving predictions by for each split on each evaluation run (rejected - better practice to store by default; can always be manually deleted if 
                    storage becomes a major concern; usually not an issue)
- **Type:** Tactical (easy to remove, but has negative implications on data analysis and auditing)

## Explainability

### Train run canonical for explainability runs
- **Decision:** Tie the explainability run to a given train run
- **Rationale:** Explainability run explains a model, so it has to be bound to a training run
- **Alternatives:** No alternatives (we need a model/pipeline as input to have something to explain)
- **Type:** Foundational (no viable alternatives if explainability is done separately from training - which is the intended architecture)

### Snapshots for explainability
- **Decision:** One trained model can be explained many times; each explainability run stores its artifacts in 
                `experiments/{problem_name}/{segment}/{model_version}/{experiment_id}/explainability/{explain_run_id}/`
- **Rationale:** In addition to the reasoning from [Snapshots for evaluation](#snapshots-for-evaluation), explain.py receives top_k as an optional cli argument (otherwise defaults to a 
                value defined in model specs), which means that different explainability runs can produce different artifacts
- **Alternatives:** Allow only one explainability run per trained model/pipeline (rejected - in addition to the reasoning from [Snapshots for evaluation](#snapshots-for-evaluation), the 
                    implied inability to use a different top_k would imply that we either have to manually alter the produced artifacts, which is unsafe and unreliable, or do another 
                    training run, which is expensive)
- **Type:** Structural (the implementation itself is not a big problem - the architectural and business-related implications are)

## Promotion

### Evaluation and explainability canonical for promotion runs
- **Decision:** Tie the promotion runs to evaluation and explainability runs from the same model (there are proper validations in place for this)
- **Rationale:** Evaluation metrics are used for decision-making, while ensuring the existence of explainability artifacts is a necessary requirement (otherwise the models cannot be used 
                as tools for an LLM, since the LLM will not be able to properly interpret the results for the end-user)
- **Alternatives:** Only tie promotion to evaluation (rejected - the end-goal is using the trained models as tools within an LLM, and the LLM needs explainability-produced artifacts to 
                    properly interpret the results); Tie to training and use training metrics for decision-making (rejected - training runs produce only basic, quick metrics, while evaluation 
                    runs produce extensive, metrics that are relevant for comparing models)
- **Type:** Foundational (technically not that difficult to implement, but antithetical to the business goals)

### Snapshots for promotion created at a shallow folder level - `model_registry/runs/{run_id}/`
- **Decision:** Store the artifacts created by promotion runs in `model_registry/runs/{run_id}/`
- **Rationale:** Most experiments end up being useless and do not reach the promotion step; run ids of relevant promotion runs are saved in the model registry or the archive, making them 
                easy to find if need be
- **Alternatives:** Nest promotion snapshots like so: `model_registry/runs/{model_problem}/{segment}/{run_id}/` (rejected - a viable idea, but adds too much complexity, considering how 
                    unlikely it is to ever be needed)
- **Type:** Structural (would need to implement branching downstream if monitoring and analytical logic gets implemented)

### Snapshots for staging and promotion live in the same folder - `model_registry/runs/{run_id}/`
- **Decision:** Treat both staging and promotion as logically equivalent by storing their artifacts in `model_registry/runs/{run_id}/`
- **Rationale:** All staged and promoted models are defined within `model_registry/models.yaml`, hence decoupling their artifact locations would be confusing and unnecessary
- **Alternatives:** Store promotion-stage runs in `model_registry/runs/promotion/{run_id}/` and staging-stage runs in `model_registry/runs/staging/{run_id}/` (rejected - could be considered 
                    confusing; adds unnecessary nesting)
- **Type:** Structural (same reasoning as in [Snapshots for promotion created at a shallow folder level](#snapshots-for-promotion-created-at-a-shallow-folder-level---model_registryrunsrun_id))

### Same script for staging, promotion and archiving
- **Decision:** Use promote.py for all staging, promotion, and archiving logic
- **Rationale:** Staging and promotion are similar, while archiving depends on promotion
- **Alternatives:** Add stage.py (rejected - unnecessary complexity; much of the code is exactly the same for staging and promotion)
- **Type:** Structural (would need to pay attention to make the implementation modular)

### Allow promotion without staging
- **Decision:** Promoting the model without previously staging it is allowed
- **Rationale:** The proper checks are still in check (candidate model needs to beat both the predefined thresholds and the current production model); staging is not always a requirement in practice
- **Alternatives:** Always require a model to be stages before allowing its promotion (rejected - a viable solution for extremely strict corporate systems, but it is currently not necessary and 
                    could add a lot of complexity, like requiring the model to be staged for at least X days before promotion)
- **Type:** Structural (staging is currently treated as an optional test-mode; the promotion logic completely ignores which model is currently staged; changing this would imply a major paradigm shift 
            in that regard)

### Only production models are archived
- **Decision:** Archive production models when new ones replace them; do not archive all of the staged models
- **Rationale:** Staging is just a testing phase, which is not that important for auditing and monitoring; keeping all of the staged models in archive can clog the document with useless information
- **Alternatives:** Keep the staged models in the archive as well (rejected - unjustified use of space)
- **Type:** Structural (adding a few lines of code is easy, but the entire archive would need to be restructured, which is problematic, since archive is the source of truth for past models)

### All discontinued production models are archived
- **Decision:** Whenever a candidate model gets promoted, the previous production model gets stores in the archive
- **Rationale:** Improves auditability, helps understand how models improved over time
- **Alternatives:** Choose whether the previous production model gets archived through a cli argument (rejected - leads to inconsistent and practically useless archive data)
- **Type:** Structural (the coding implementation is not necessarily challenging, but the business implications are significant)

### All current staging and production models live in `model_registry/models.yaml` and are nested by problem type and segment
- **Decision:** Keep all of the staging and production models within the `model_registry/models.yaml` file and nest them by problem type and segment
- **Rationale:** Easy to visually scan through all of the currently staged/promoted models; makes inference much easier to implement
- **Alternatives:** Store staging models separately (rejected - adds an unnecessary level of complexity); store based on problem type and segment (rejected - impractical and unprofessional)
- **Type:** Structural

### A single archive for all models, internal nesting by problem type and segment
- **Decision:** Archive all of the previous production models within `model_registry/archive.yaml` and nest them by problem type and segment
- **Rationale:** Archiving is mostly done for auditing and data analysis - this architecture fits that purpose perfectly
- **Alternatives:** Use one archive but no nesting (rejected - it is easier to get the desired models when nesting is in place, and it doesn't incur any extra cost)
- **Type:** Structural

## Post-promotion

### Simple inference and monitoring
- **Decision:** Implement relatively simple inference and monitoring logic
- **Rationale:** Complex logic not needed at this stage, but some logic is desirable to complete the lifecycle
- **Alternatives:** Implement complex logic immediately to avoid future refactoring (rejected - too much effort for something that is currently of little use);
                    Do not implement any inference and/or monitoring logic (rejected - a simple implementation does not consume too much time, while logically completing the model lifecycle)
- **Type:** Structural

### Each inference run stores artifacts in its own subdirectory
- **Decision:** Store predictions and metadata of each inference run in its own subdirectory
- **Rationale:** Easier upstream implementation (monitoring)
- **Alternatives:** Group predictions by hour and/or day, while preventing redundant predictions (same rows) (rejected - 
                    inference run are not common enough at this stage to demand this level of code sophistication)
- **Type:** Structural

### Hardcoding of predictions schema
- **Decision:** Keep the predictions schemas versioned, but hardcoded in a python file
- **Rationale:** Not expected to change very often; avoids additional yaml configs
- **Alternatives:** Implement configs for schema versioning (rejected - would require additional validations and similar complexity, while not needed);
                    Do not version schemas at all (rejected - hardcoded versioning is easy to implement, but still fairly useful; can be tracked via git diffs)
- **Type:** Tactical

### Hardcoding of feature drift thresholds
- **Decision:** Use the same hardcoded thresholds to track feature drifting in the monitoring pipeline
- **Rationale:** Separate thresholds per feature are rarely needed; the current implementation only logs warnings and errors anyway - no code blocking logic that
                would make the thresholds really matter
- **Alternatives:** Implement configs specifying thresholds per feature or feature set (rejected - can quickly turn into a huge file that requires lots of maintenance
                    and validation, but provides very little benefit)
- **Type:** Structural

### No execution blocking on feature drifting and performance degradation
- **Decision:** Do not block the code execution if any level of feature drifting or model performance degradation is detected
- **Rationale:** The code already logs warnings and saves the monitoring report - the user should decide what to do with that on their own
- **Alternatives:** Fail loudly if major drifts or performance degradation are detected (rejected - this implementation would be too 
                    aggressive; instead, the user should be informed, but allowed to decide what to do with that information on their own)
- **Type:** Tactical

### Allowing of different schemas for production and staging models
- **Decision:** Allow the production and staging models to use different schemas
- **Rationale:** The models should operate within the same segment and problem type, but the schemas should be allowed to evolve -
                this allows the models to gradually become better, rather than forcing a biased structure for little benefit
- **Alternatives:** Force the same schema in order to enable better analysis (rejected - while analysis could indeed be deeper
                    with this implementation, the lack of flexibility would actually make the models perform worse in the long run);
                    Nest the models by version in promotion as well, and then do the same in post-promotion pipelines (rejected -
                    this would make the implementation significantly more complex with very little benefit, unless this scales considerably)
- **Type:** Foundational

### Comparison of the production and staging models in monitoring runs
- **Decision:** Compare the production and staging models in monitoring runs
- **Rationale:** This relatively simple implementation allows for easier comparison of the two models, instead of having to manually
                calculate the differences in performance
- **Alternatives:** Only store individual model performances in the monitoring report (rejected - the storage is not a major issue, while
                    the implementation of this logic is fairly useful)
- **Type:** Tactical

### Lack of monitoring report validation
- **Decision:** Unlike all of the other json files, do not validate the monitoring report
- **Rationale:** The report has very few stable elements, and varies greatly based on whether only one model type is present
                (production/staging) or both; this makes the implementation difficult, while providing little benefit;
                there is no upstream code from here, and the report is expected to be viewed immediately and stop
                being useful after a short-to-moderate period of time anyway
- **Alternatives:** Validate the monitoring report anyway (rejected - would add complexity and possibly even false confidence,
                    since 100% accurate validation is unlikely, given the current implementation)
- **Type:** Tactical

### Assumption of proba_1 being the positive class
- **Decision:** Assume proba_1 is the positive class for classification tasks
- **Rationale:** Works well for binary classification tasks; current implementation does not include any multi-class problems anyway
- **Alternatives:** Prepare the code for multi-class problems immediately (rejected - would add lots of abstraction with no immediate benefit)
- **Type:** Tactical

## ML Service

### ML service existence
- **Decision:** Create an ML service to simplify the workflow
- **Rationale:** Easier to operate the ml workflow, better UX, less error-prone
- **Alternatives:** Stick to CLI-only (rejected - dashboards are convenient and not too difficult to create and maintain)
- **Type:** Convenience (eveything still works fine without this code)

### Inclusion of docs in the service
- **Decision:** Enable documentation reading in the service, even if not perfect
- **Rationale:** A simple implementation that reduces the need to leave the service
- **Alternatives:** Keep the service for pipelines, scripts and configs only (rejected - no reason to do so)
- **Type:** Convenience

### Inclusion of file and directory structure viewing within the service
- **Decision:** Allow the yaml and json files to be viewed, and the directory trees to be shown within the service
- **Rationale:** Enables viewing the newly created artifacts, as well as understanding the directory structure - all
                using a simple UI
- **Type:** Convenience

### No log streaming
- **Decision:** Do not enable log streaming within the service
- **Rationale:** This would immediately increase the complexity considerably, with no apparent reason (the message
                already shows the important logs upon completion, while logs can easily be viewed manually during the
                execution if needed)
- **Alternatives:** Enable log streaming (rejected - too complex, while adding very little value at the current stage)
- **Type:** Structural

### No user registration and login
- **Decision:** Do not support user registration and login
- **Rationale:** The current app is not meant to be used by more than a handful of users
- **Alternatives:** Implement simple registration and login (rejected - the only benefit would be automatic lineage 
                    created_by element writing - not worth it)
- **Type:** Tactical

### Display of the important logs and prints upon completion
- **Decision:** Display the relevant logs and prints upon completion
- **Rationale:** Allows for a quick inspection of what the run did
- **Alternatives:** Only notify the user on the success status (rejected - the user should be able to quickly check
                    some of the important bits of information regarding the script execution without leaving the service)
- **Type:** Convenience