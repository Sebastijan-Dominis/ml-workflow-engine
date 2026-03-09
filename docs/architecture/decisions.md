# Architectural Decisions

This file records key architectural decisions, their rationale, and alternatives considered.

## General/global

### Separation of `ml` from `pipelines`
- **Decision:** Keep ML logic independent from pipelines
- **Rationale:** Promotes reusability and testability
- **Alternatives:** Integrate ML logic into pipelines (rejected - less reusable)

### Logs stored at the lowest level
- **Decision:** Store logs in the lowest-level folder that was created by a given run of a pipeline
- **Rationale:** Easy to understand what the logs refer to, easy to navigate the logs
- **Alternatives:** Create a logs folder at the repo root level, then branch from there (rejected - mostly harmless, but unnecessary and harder to navigate and maintain); log to the same location depending on the pipeline (e.g. all calls to `freeze.py` log to `freeze.log`) (rejected - quickly leads to logging hell and extremely hard to navigate)

### Defining the logging level through a cli argument
- **Decision:** Define the logging level (DEBUG/INFO...) through a cli argument for each given pipeline
- **Rationale:** Flexible, with no clear downsides (e.g. use DEBUG for development and INFO for production)
- **Alternatives:** Always use the same logging level (rejected - very inflexible); Define the logging level in configs (rejected - unnecessarily complicated; logging should not depend on configs, and should be more flexible)

### Decoupling of feature sets and data
- **Decision:** Keep feature sets and data decoupled, even though features depend on processed datasets
- **Rationale:** Allows for feature sets to be composed of features from various datasets
- **Alternatives:** Always use the same dataset for the same feature set (rejected - feature in different datasets may be logically similar; less flexible)

### Decoupling of modeling and feature freezing
- **Decision:** Decouple feature freezing from modeling (search, runners)
- **Rationale:** Makes the features model-agnostic and reusable, allows for different models and their versions to use different feature sets, very scalable
- **Alternatives:** Tie features to each model (rejected - less scalability, flexibility and reusability)

## Artifacts and configs - general

### Informational artifacts stored in `json` format
- **Decision:** Store all of the informational artifacts (metadata, runtime info, metrics) in `json` format
- **Rationale:** Easy to validate, read and work with
- **Alternatives:** Use yaml instead (rejected - json is more readable and often considered standard)

### Configs stored in `yaml` format
- **Decision:** Store all of the configs in `yaml` format
- **Rationale:** Easy to validate, read and work with
- **Alternatives:** No alternatives were considered for this decision

### Configs stored in the `configs/` folder
- **Decision:** Store all of the configs within the `configs/` folder, with relevant, semantic nesting
- **Rationale:** Easy to access, work with, and reason about
- **Alternatives:** Use sql instead (rejected - a viable alternative at great scale, but would add unnecessary complexity to the current system; migrating to it later on is simple enough)

### Experiments and all of their artifacts stored in the `experiments/` folder
- **Decision:** Store all of the experiments and the artifacts they produce (e.g. metadata, runtime info, metrics, models, pipelines) within the `experiments/` folder
- **Rationale:** Easy to access, work with, and reason about
- **Alternatives:** Store models and pipelines separately (rejected - models and pipelines are strongly tied to training runs, and decoupling them would decrease safety and add complexity); use cloud services (rejected - great option long-term, but would incur unnecessary costs at the current stage)

### Metadata produced by each pipeline (e.g. `freeze.py`)
- **Decision:** Every pipeline produces some metadata
- **Rationale:** Forces good practices, like auditability
- **Alternatives:** Only produce metadata when expected to be used by downstream code (rejected - considered to be a bad practice)

### Storing of runtime information for feature freezing and experiments
- **Decision:** Store runtime information for each run of `freeze.py`, `search.py`, `train.py`, `evaluate.py` and `explain.py` (`freeze.py` includes it in metadata, while the other four produce a separate artifact)
- **Rationale:** Usually cheap to generate and store; improves auditability and reproducibility
- **Alternatives:** Do not store runtime information (rejected - the benefits outweight the costs)

## Modeling

### Models defined by problem type, segment and version
- **Decision:** Define the models by the problem they solve, the segment they encompass, and the version of those two (based on configs)
- **Rationale:** Allows for seamless navigation and scalability, very future-proof
- **Alternatives:** Define only by problem type or problem type and segment (rejected - less scalable, harder to navigate); Define by algorithm as well (rejected - unnecessarily complicated)

### Nesting search and training configs same as model specs
- **Decision:** Use the same folder nesting for search and training configs as with model specs configs
- **Rationale:** Easy to navigate, seamless config resolution at runtime
- **Alternatives:** Version search/train configs separately (rejected - unnecessary complexity)

### Search/training final configs overriding order: global defaults -> algorithm defaults -> model specs -> search/train -> env (optional) 
- **Decision:** When forming the final configs for search or training, first override the global defaults by algorithm defaults, then by model specs, then by search or training configs, and finally by environment configs (optional)
- **Rationale:** Defaults are always good to have for future-proofing, model specs define most of the behavior, search and training configs define information specific to their stage of modeling, environment configs allow easy override of the most impactful elements (e.g. number of iterations); more reusable and less error prone when search and training use the same logic
- **Alternatives:** Do not include defaults and/or environment configs (rejected - less scalable; those configs are extremely inexpensive to store, but can easily come in handy); decouple search from training config creation (rejected - more error prone, more boilerplate code)

### Passing of env as a cli argument
- **Decision:** Pass environment as a cli argument in relevant pipelines
- **Rationale:** Dev and test allow for quickly iterating and testing the pipelines, while prod ensures the required quality in production - more flexible to use through cli
- **Alternatives:** Do not use env at all (rejected - env override is extremely useful)

### Feature engineering decided at feature freezing
- **Decision:** Decide which engineered features to use for a given version of a given feature set through the feature registry
- **Rationale:** Ensures that the input features necessary to create the engineered/derives features are always present by raising an error early (during feature freezing) if they are not
- **Alternatives:** Engineer features during data preprocessing (rejected - could lead to a bad downstream situation where a model's pipeline wants to create a feature, but does not have the necessary input - e.g. it wants to create adr_per_person, but adr is not present)

### Operators defined by derived schemas
- **Decision:** Create operators (==engineer features) at runtime from derived schemas
- **Rationale:** Ensures that the input features necessary to create the engineered features are always present, given that the derived schemas are created programmatically by `freeze.py`
- **Alternatives:** Manually define operators via configs (rejected - look at [Feature engineering decided at feature freezing](#feature-engineering-decided-at-feature-freezing))

### Target creation at runtime
- **Decision:** Create the target variable at runtime
- **Rationale:** Enables seamless versioning of target creation; simplifies validation
- **Alternatives:** Freeze the target variable during the freezing step (rejected - target creation may require features from different datasets; makes versioning and validation more challenging)

### Target versioning
- **Decision:** Version how the target variable is created
- **Rationale:** Although rarely, we may sometimes want to create the target for a new version of the same problem type and segment in a different way
- **Alternatives:** Always create the target for the same model in the same way (rejected - less flexible and future-proof)

## Orchestration

### Running all data preprocessing (raw/interim/processed) with a single orchestrator enabled
- **Decision:** Enable the execution of all data preprocessing for all of the available datasets and configs (the orchestrator figures it out)
- **Rationale:** Drammatically speeds up the data preprocessing step when needed
- **Alternatives:** Only ever use individual pipelines (e.g. build_interim_dataset.py) (rejected - can be time-inefficient; proper validations are in check anyway)

### Freezing all feature sets from the feature registry with a single orchestrator enabled
- **Decision:** Enable the freezing of all feature sets found in the feature registry by running a single orchestrator
- **Rationale:** More efficient when needed
- **Alternatives:** Force freezing feature sets individually (rejected - can be time-inefficient; proper validations are in check anyway)

### Executing the entire experiment for a single model with a single orchestrator enabled
- **Decision:** Enable a sequential execution of search, training, evaluation and explainability for a given model by defaulting to the latest snapshots for training, evaluation and explainability steps
- **Rationale:** More efficient when needed
- **Alternatives:** Force running the pipelines individually (rejected - the runs log which experiment id and training id they defaulted to anyway, so tracking it is still straightforward)

### Executing all workflows with a single orchestrator enabled
- **Decision:** Enable running all data preprocessing for all of the available datasets and configs, followed by freezing of all feature sets found in the feature registry, followed by the execution of search, training, evaluation and explainability for all of the available models (orchestrator figures it out by inspecting how `configs/model_specs/` is nested) by defaulting to latest snapshots for training, evaluation and explainability steps
- **Rationale:** Extremely efficient when appropriate
- **Alternatives:** Forbid such high-level orchestration (rejected - the user is responsible for what happens next; proper validations are in check and the logs allow for traceability)

### Orchestrating promotion disabled
- **Decision:** Do not orchestrate the promotion step
- **Rationale:** Hard to implement if meant to be run with the grand orchestrator; easy to result in unexpected outcomes; usually not needed
- **Alternatives:** Create `stage_all.py` or `promote_all.py` and run it right after `execute_all_experiments_with_latest.py` (rejected - hard to implement, as we don't know which experiment/training/evaluation/explainability ids to use prior to running the grand orchestrator; rarely needed; could end up polluting the model registry, which is a terrible idea, as that is the source of truth for staging and production models)

## Data

### Raw data versioning
- **Decision:** Use versioning for raw data
- **Rationale:** Adding or removing columns within the same dataset simply implies a new version - logically it is the same dataset
- **Alternatives:** Imply a new version in the name (e.g. hotel_bookings_v2) (rejected - semantically complicated)

### Snapshots for raw data
- **Decision:** Use snapshots for updated data from the same dataset and version
- **Rationale:** Incoming new data simply implies a new snapshot - logically it is the same version of the same raw data
- **Alternatives:** Overwrite the older snapshot and always use the newest data (rejected - harms reproducibility, terrible for auditing); Add timestamps to the name (e.g. hotel_bookings/v1_{timestamp}) (rejected - extremely complicated with no clear upside)

### Separation of raw and interim data
- **Decision:** Keep interim data separately from raw
- **Rationale:** Raw data is the source of truth. Interim optimizes for efficiency, while preserving the source of truth.
- **Alternatives:** Optimize and overwrite raw data immediately (rejected - loses the source of truth)

### Interim datasets versioning
- **Decision:** Use versioning for interim datasets
- **Rationale:** Optimizing raw data by opting for different data types simply implies a new version - logically tied to the same version of the same raw data
- **Alternatives:** Always use the same optimization for each version of each raw data (rejected - less flexible with no clear upsides); Imply new version in the name (e.g. hotel_bookings_v2) (rejected - semantically complicated)

### Snapshots for interim datasets
- **Decision:** Use snapshots for updated interim datasets of the same version
- **Rationale:** Incoming new data simply implies a new snapshot - logically it is the same version of the same interim dataset
- **Alternatives:** same as in [Use snapshots for raw data](#use-snapshots-for-raw-data)

### Separation of interim and processed data
- **Decision:** Keep processed data separately from interim
- **Rationale:** Interim is used for memory optimization, while processed takes care of the necessary transformations.
- **Alternatives:** Optimize and process in the same step (rejected - we often want to process differently, but optimize in the same way)

### Processed datasets versioning
- **Decision:** Use versioning for processed datasets
- **Rationale:** Applying different preprocessing simply implies a new version - logically tied to the same version of the same interim dataset
- **Alternatives:** Always preprocess a given version of an interim dataset in the same way (rejected - extremely unflexible and impractical); Imply new version in the name (e.g. hotel_bookings_v2) (rejected - semantically complicated)

### Snapshots for processed datasets
- **Decision:** Use snapshots for updated data processed datasets of the same version
- **Rationale:** Incoming new data simply implies a new snapshot - logically it is the same version of the same processed dataset
- **Alternatives:** same as in [Snapshots for raw data](#snapshots-for-raw-data)

## Features

### Division into feature sets
- **Decision:** Create feature sets by combining only the logically similar features
- **Rationale:** Promotes explainability, reusability, and scalability
- **Alternatives:** Use generic "features" with many different versions (rejected - quickly leads to versioning hell, hard to interpret)

### Feature set versioning
- **Decision:** Version feature sets
- **Rationale:** Quick and easy adaptation to new modeling needs
- **Alternatives:** Always use the same set of features in a feature sets (rejected - extremely bad practice, since different models often imply different feature needs)

### Snapshots for feature sets
- **Decision:** Use snapshots for freezing feature sets of the same version with updated data from processed datasets
- **Rationale:** Allows for a seamless creation of the same sets of features with new rows, while also preserving the older snapshots for auditing an reproducibility
- **Alternatives:** same as in [Snapshots for raw data](#snapshots-for-raw-data)

### Keeping of the row_id in frozen features
- **Decision:** Keep row_id column when freezing any given feature set
- **Rationale:** Allows for seamless merging of many feature sets at runtime, regardless of their respective row counts
- **Alternatives:** Drop row_id while freezing a feature set (rejected - makes proper merging at runtime impossible)

### Saving of input and derived schemas at the version level
- **Decision:** Save both input and derived schemas once at the version level (one for all snapshots)
- **Rationale:** Schemas depend on configs, which should not change across runs; more logical and saves memory
- **Alternatives:** Save new schemas for each snapshot (rejected - redundant, could be more error prone if the code did not include all of the validations that it already does)

### The disinclusion of row_id in input_schema
- **Decision:** Do not include row_id in feature set's input schema
- **Rationale:** Row id is only used for merging, should never reach the model, and does not provide any value outside of merging; this decision aligns well with the rest of the code logic
- **Alternatives:** Include row_id in input schema (rejected - it is only a technical feature)

### Not saving derived schema when no operators in configs
- **Decision:** Do not save a derived schema if no operators are being created, meaning all of the features are already included in the input schema
- **Rationale:** Avoids flooding the feature store with empty derived schemas
- **Alternatives:** Save an empty derived schema instead (rejected - harmless, but unnecessary)

## Search

### One broad and one optional narrow search
- **Decision:** Always do one broad hyperparameter search, optionally followed by one narrow search
- **Rationale:** Flexible enough to only do one search for models that do not require more than that; there is no need for more than two searches - increasing iterations is a better approach
- **Alternatives:** Allow the configs to define the number of searches (rejected - unnecessary; just increase iterations or improve data if results are subpar); always do both broad and narrow search (rejected - some models converge very quickly and doing two searches is a waste of time and resources in that case)

### Forced use of randomized search
- **Decision:** Always do a randomized search with RandomizedSearchCV
- **Rationale:** It is efficient and easy to use
- **Alternatives:** Use grid search for narrow search (rejected - grid search is extremely slow and usually does not lead to notably better results); Allow configs to define which search method will be implemented (rejected - randomized search always works best); Add grid search as the third search (argument logically connected to [One broad and one optional narrow search](#one-broad-and-one-optional-narrow-search)) (rejected - extremely inefficient)

### Parameter distribution for broad search defined in configs
- **Decision:** Define the parameter distributions in search configs
- **Rationale:** Provides flexibility, especially when we have an idea of where a given hyperparameter could end up; scalable
- **Alternatives:** Define parameter distributions in code, and use the same ones with all models (rejected - less efficient, less flexible)

### Parameter configurations for narrow search defined in configs
- **Decision:** Define the configurations that take the best hyperparameters from broad search at runtime and create parameter distrubutions for narrow search in search configs
- **Rationale:** Same as in [Parameter distribution for broad search defined in configs](#parameter-distribution-for-broad-search-defined-in-configs)
- **Alternatives:** Always use the same numbers to calculate parameter distributions for the narrow search (rejected - less efficient, less flexible)

### Saving of best hyperparameters from broad and narrow search respectively at runtime
- **Decision:** Save the best hyperparameters upon finishing both broad and narrow search respectively, at runtime; saved in the failure management folder
- **Rationale:** Allows to seamlessly continue with either narrow search or persistence in case the code fails at runtime, or the hardware running it suddenly stops working
- **Alternatives:** Do not save anything when running the search pipeline (rejected - search is expensive, failures with no backup can cost many dollars and hours)

### Allowing deletion of the saved best hyperparameters from broad and narrow search
- **Decision:** Enable the saved best hyperparamters from broad and narrow search to be dynamically deleted if the search run finishes successfully, but do not force the deletion
- **Rationale:** Prevents the hoarding of useless information without requiring manual deletion; flexible to allow override (skipping deletion) when desired
- **Alternatives:** Do not delete the saved best hyperparameters from broad and narrow search dynamically (rejected - can easily fill space with useless data, and requires unnecessary manual intervention); always delete (rejected - less flexible)

### Search runs create experiments
- **Decision:** Each search run creates a new experiment
- **Rationale:** Experiments include various artifacts from search and runners, all of which are ultimately defined by the search; hence new search = new experiment
- **Alternatives:** Enable multiple search runs per experiment (rejected - this is often done in research-focused ml-infra, since running the same code with the same configs on slightly different hardware may result in slightly different floating points - this is a product-oriented system, so those tiny differences are considered to be irrelevant)

### Search run artifacts stored in `experiments/.../{experiment_id}/search/`
- **Decision:** Store all of the artifacts generated by a search run in `experiments/{problem_name}/{segment}/{model_version}/{experiment_id}/search/`
- **Rationale:** Search is canonical for a given experiment, so there is no need for snapshots; nevertheless, including it within a nested search folder makes it semantically clear that the artifacts were generated by the search
- **Alternatives:** Create a nested search run for storing artifacts created by a search run (rejected - reasoning explained already); Store the generated artifacts at the experiment folder level (rejected - viable alternatives, but less semantically accurate)

## Training

### Experiment (created by a search run) canonical for training runs
- **Decision:** Tie the training run to a given experiment (generated by a search run)
- **Rationale:** Training needs the best parameters found from search to assemble the final runtime configs; results will greatly differ based on that
- **Alternatives:** Decouple training from search (rejected - training without using the best params from any given search run is useless)

### Snapshots for training
- **Decision:** One experiment can have many training runs; each stores its artifacts in `experiments/{problem_name}/{segment}/{model_version}/{experiment_id}/training/{train_run_id}/`
- **Rationale:** Training with the same configs and hardware can result in a different model if we train on updated data; this prevents wasting resources on re-searching params when that is not necessary
- **Alternatives:** One training run per experiment (rejected - requiring each training run to imply a new search run is extremely inefficient)

### Saving model snapshots as backup while training
- **Decision:** Save a model snapshot every x seconds (defined by configs) while the training is being executed
- **Rationale:** In case of a system failure, we can continue training from where we left off, rather than start from the start (e.g. we can continue from iteration 987, rather than 0)
- **Alternatives:** Not saving anything during training (rejected - training can be expensive, and relying on the belief that the execution will surely go well every time is dangerous)

### Allowing deletion of the training backups
- **Decision:** Enable the training backups (model snapshots) to be dynamically deleted if the training run finishes successfully, but do not force the deletion
- **Rationale:** Same as [Allowing deletion of the saved best hyperparameters from broad and narrow search](#allowing-deletion-of-the-saved-best-hyperparameters-from-broad-and-narrow-search)
- **Alternatives:** Same as [Allowing deletion of the saved best hyperparameters from broad and narrow search](#allowing-deletion-of-the-saved-best-hyperparameters-from-broad-and-narrow-search)

## Evaluation

### Train run canonical for evaluation runs
- **Decision:** Tie the evaluation run to a given train run
- **Rationale:** Evaluation run evaluates a model, so it has to be bound to a training run
- **Alternatives:** No alternatives (we need a model/pipeline as input to have something to evaluate)

### Snapshots for evaluation
- **Decision:** One trained model can be evaluated many times; each evaluation run stores its artifacts in `experiments/{problem_name}/{segment}/{model_version}/{experiment_id}/evaluation/{eval_run_id}/`
- **Rationale:** While usually the same, evaluation runs can sometimes result in different output (f.x. if we change the code - in that case the warning is logged, but the execution is allowed to continue); this approach prevents the expensive re-training, and allows us to seamlessly re-evaluate our older models with new metrics
- **Alternatives:** Allow only one evaluation run per trained model/pipeline (rejected - dangerous, since adding new metrics in the future may result in hundreds of models requiring another training run)

### Storing row-by-row predictions by split (train/val/test)
- **Decision:** Store the exact row-by-row predictions the model made for each split (train/val/test)
- **Rationale:** In a case of weird outcomes, this provides an easy way of inspecting where the model is failing; could potentially be re-used in explainability runs (currently not implemented)
- **Alternatives:** Save storage space by not saving predictions by for each split on each evaluation run (rejected - better practice to store by default; can always be manually deleted if storage becomes a major concern; usually not an issue)

## Explainability

### Train run canonical for explainability runs
- **Decision:** Tie the explainability run to a given train run
- **Rationale:** Explainability run explains a model, so it has to be bound to a training run
- **Alternatives:** No alternatives (we need a model/pipeline as input to have something to explain)

### Snapshots for explainability
- **Decision:** One trained model can be explained many times; each explainability run stores its artifacts in `experiments/{problem_name}/{segment}/{model_version}/{experiment_id}/explainability/{explain_run_id}/`
- **Rationale:** In addition to the reasoning from [Snapshots for evaluation](#snapshots-for-evaluation), explain.py receives top_k as an optional cli argument (otherwise defaults to a value defined in model specs), which means that different explainability runs can produce different artifacts
- **Alternatives:** Allow only one explainability run per trained model/pipeline (rejected - in addition to the reasoning from [Snapshots for evaluation](#snapshots-for-evaluation), the implied inability to use a different top_k would imply that we either have to manually alter the produced artifacts, which is unsafe and unreliable, or do another training run, which is expensive)

## Promotion

### Evaluation and explainability canonical for promotion runs
- **Decision:** Tie the promotion runs to evaluation and explainability runs from the same model (there are proper validations in place for this)
- **Rationale:** Evaluation metrics are used for decision-making, while ensuring the existence of explainability artifacts is a necessary requirement (otherwise the models cannot be used as tools for an LLM, since the LLM will not be able to properly interpret the results for the end-user)
- **Alternatives:** Only tie promotion to evaluation (rejected - the end-goal is using the trained models as tools within an LLM, and the LLM needs explainability-produced artifacts to properly interpret the results); Tie to training and use training metrics for decision-making (rejected - training runs produce only basic, quick metrics, while evaluation runs produce extensive, metrics that are relevant for comparing models)

### Snapshots for promotion created at a shallow folder level - `model_registry/runs/{run_id}/`
- **Decision:** Store the artifacts created by promotion runs in `model_registry/runs/{run_id}/`
- **Rationale:** Most experiments end up being useless and do not reach the promotion step; run ids of relevant promotion runs are saved in the model registry or the archive, making them easy to find if need be
- **Alternatives:** Nest promotion snapshots like so: `model_registry/runs/{model_problem}/{segment}/{run_id}/` (rejected - a viable idea, but adds too much complexity, considering how unlikely it is to ever be needed)

### Snapshots for staging and promotion live in the same folder - `model_registry/runs/{run_id}/`
- **Decision:** Treat both staging and promotion as logically equivalent by storing their artifacts in `model_registry/runs/{run_id}/`
- **Rationale:** All staged and promoted models are defined within `model_registry/models.yaml`, hence decoupling their artifact locations would be confusing and unnecessary
- **Alternatives:** Store promotion-stage runs in `model_registry/runs/promotion/{run_id}/` and staging-stage runs in `model_registry/runs/staging/{run_id}/` (rejected - could be considered confusing; adds unnecessary nesting)

### Same script for staging, promotion and archiving
- **Decision:** Use promote.py for all staging, promotion, and archiving logic
- **Rationale:** Staging and promotion are similar, while archiving depends on promotion
- **Alternatives:** Add stage.py (rejected - unnecessary complexity; much of the code is exactly the same for staging and promotion)

### Enabled promotion without staging
- **Decision:** Promoting the model without previously staging it is enabled
- **Rationale:** The proper checks are still in check (candidate model needs to beat both the predefined thresholds and the current production model); staging is not always a requirement in practice
- **Alternatives:** Always require a model to be stages before allowing its promotion (rejected - a viable solution for extremely strict corporate systems, but it could add a lot of complexity, like requiring the model to be staged for at least X days before promotion, which is usually completely unnecessary)

### Only production models are archived
- **Decision:** Archive production models when new ones replace them; do not archive all of the staged models
- **Rationale:** Staging is just a testing phase, which is not that important for auditing and monitoring; keeping all of the staged models in archive can clog the document with useless information
- **Alternatives:** Keep the staged models in the archive as well (rejected - unjustified use of space)

### All discontinued production models are archived
- **Decision:** Whenever a candidate model gets promoted, the previous production model gets stores in the archive
- **Rationale:** Improves auditability, helps understand how models improved over time
- **Alternatives:** Choose whether the previous production model gets archived through a cli argument (rejected - leads to inconsistent and practically useless archive data)

### All current staging and production models live in `model_registry/models.yaml` and are nested by problem type and segment
- **Decision:** Keep all of the staging and production models within the `model_registry/models.yaml` file and nest them by problem type and segment
- **Rationale:** Easy to visually scan through all of the currently staged/promoted models; makes inference much easier to implement
- **Alternatives:** Store staging models separately (rejected - adds an unnecessary level of complexity); store based on problem type and segment (rejected - impractical and unprofessional)

### A single archive for all models, internal nesting by problem type and segment
- **Decision:** Archive all of the previous production models within `model_registry/archive.yaml` and nest them by problem type and segment
- **Rationale:** Archiving is mostly done for auditing and data analysis - this architecture fits that purpose perfectly
- **Alternative:** Use one archive but no nesting (rejected - it is easier to get the desired models when nesting is in place, and it doesn't incur any extra cost)