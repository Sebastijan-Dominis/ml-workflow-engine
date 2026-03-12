# Maintenance Guide

Instructions for maintaining and troubleshooting the project.

## Monitoring
- Monitor your pipeline runs by reading the logs.
- All of the logs belonging to any individual pipeline are stored in the same folder as the artifacts that the pipeline produces.
    > For instance, `pipelines/runners/train.py` logs go to 
    `experiments/{problem_type}/{segment}/{model_version}/{experiment_id}/training/{train_run_id}/train.log`
- All of the orchestration pipelines' logs can be found within `orchestration_logs/` in locations that are 
    logically easy to find.
    > For instance, `pipelines/orchestration/experiments/execute_experiment_with_latest.py` logs to 
    `orchestration_logs/experiments/execute_experiment_with_latest/{run_id}/experiment_execution.log`
- Each pipeline contains only the information relevant to its own logic.
    > Orchestrators log only high-level information -> go to individual pipeline logs for more details
- Logging level for each pipeline run can be set using a cli argument
    > Default is `INFO`
    > Make sure to write the argument in uppercase letters
- The meaning of error codes:
    - `0` -> `success` All went well.
    - `1` -> `unexpected error` Something went wrong, but was not properly captured by the existing code.
    - `2` -> `config error` or `user error` -> Invalid or inconsistent configuration; user configuration or misuse.
    - `3` -> `data error` -> Feature store or data issues.
    - `4` -> `pipeline error` -> Violations of structural or logical expectations between pipeline
                                stages or experiment components, including incompatible artifacts,
                                lineage inconsistencies, incorrect stage ordering, or execution under
                                an unrelated or incompatible experiment context.
    - `5` -> `search error` -> Hyperparameter search failure.
    - `6` -> `training error` -> Model training failure.
    - `7` -> `evaluation error` -> Evaluation or metric computation failure.
    - `8` -> `explainability error` -> Explainability or interpretation stage failure.
    - `9` -> `persistence error` -> Experiment or artifact saving failure.

## Troubleshooting
- Check logs and error messages.
- Refer to other relevant documents, most notably [architecture overview](architecture/overview.md) and [decisions](architecture/decisions.md), for design rationale.

## Best Practices
- If an artifact gets corrupted, delete the entire snapshot and re-run a pipeline.
- If you use experiment-related orchestrators (or the master orchestrator), make sure to check their respected logs for reassurance on what happened.
    - Experiment orchestrators default to using the latest `experiment_id` for runners, and the latest `train_run_id` for evaluation and explainability runs.
    - Depending on your workflow, that may or may not result in unexpected behavior.
    - Experiment-related orchestrators are those found within `pipelines/orchestration/experiments`.
    - Master orchestrator is found within `pipelines/orchestration/master`.
- Avoid using `DEBUG` logging level unless needed.
    - The logs are quite extensive on that level, and useless in a normal workflow.
    - Logging on this level can result in unnecessary memory occupation.
    - It can be useful for troubleshooting.
- Create quality tests for any new code, as noted in [testing.md](testing.md).
    - CI will fail if less than 90% of the code is covered.
- Be careful if switching to newer python versions.
    - Many ml-related packages do not work well in newer versions.
- Prefer using conda and installing the ds-and-ml-related packages with conda-forge.