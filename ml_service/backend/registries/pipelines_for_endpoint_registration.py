from ml_service.backend.models.pipelines import (
    BuildInterimDatasetInput,
    BuildProcessedDatasetInput,
    EvaluateInput,
    ExecuteAllDataPreprocessingInput,
    ExecuteAllExperimentsWithLatestInput,
    ExecuteExperimentWithLatestInput,
    ExplainInput,
    FreezeAllFeatureSetsInput,
    FreezeFeaturesInput,
    PromoteInput,
    RegisterRawSnapshotInput,
    RunAllWorkflowsInput,
    SearchInput,
    TrainInput,
)

PIPELINES_FOR_ENDPOINT_REGISTRATION = [
    {
        "name": "register_raw_snapshot",
        "module_path": "pipelines.data.register_raw_snapshot",
        "args_schema": RegisterRawSnapshotInput,
        "boolean_args": []
    },
    {
        "name": "build_interim_dataset",
        "module_path": "pipelines.data.build_interim_dataset",
        "args_schema": BuildInterimDatasetInput,
        "boolean_args": []
    },
    {
        "name": "build_processed_dataset",
        "module_path": "pipelines.data.build_processed_dataset",
        "args_schema": BuildProcessedDatasetInput,
        "boolean_args": []
    },
    {
        "name": "freeze_feature_set",
        "module_path": "pipelines.features.freeze",
        "args_schema": FreezeFeaturesInput,
        "boolean_args": []
    },
    {
        "name": "search",
        "module_path": "pipelines.search.search",
        "args_schema": SearchInput,
        "boolean_args": ["strict", "clean_up_failure_management", "overwrite_existing"]
    },
    {
        "name": "train",
        "module_path": "pipelines.runners.training.train",
        "args_schema": TrainInput,
        "boolean_args": ["strict", "clean_up_failure_management", "overwrite_existing"]
    },
    {
        "name": "evaluate",
        "module_path": "pipelines.runners.evaluation.evaluate",
        "args_schema": EvaluateInput,
        "boolean_args": ["strict"]
    },
    {
        "name": "explain",
        "module_path": "pipelines.runners.explainability.explain",
        "args_schema": ExplainInput,
        "boolean_args": ["strict"]
    },
    {
        "name": "promote",
        "module_path": "pipelines.promotion.promote",
        "args_schema": PromoteInput,
        "boolean_args": []
    },
    {
        "name": "execute_all_data_preprocessing",
        "module_path": "pipelines.orchestration.data.execute_all_data_preprocessing",
        "args_schema": ExecuteAllDataPreprocessingInput,
        "boolean_args": []
    },
    {
        "name": "execute_experiment_with_latest",
        "module_path": "pipelines.orchestration.experiments.execute_experiment_with_latest",
        "args_schema": ExecuteExperimentWithLatestInput,
        "boolean_args": ["strict", "clean_up_failure_management", "overwrite_existing"]
    },
    {
        "name": "execute_all_experiments_with_latest",
        "module_path": "pipelines.orchestration.experiments.execute_all_experiments_with_latest",
        "args_schema": ExecuteAllExperimentsWithLatestInput,
        "boolean_args": ["strict", "clean_up_failure_management", "overwrite_existing"]
    },
    {
        "name": "freeze_all_feature_sets",
        "module_path": "pipelines.orchestration.features.freeze_all_feature_sets",
        "args_schema": FreezeAllFeatureSetsInput,
        "boolean_args": []
    },
    {
        "name": "run_all_workflows",
        "module_path": "pipelines.orchestration.master.run_all_workflows",
        "args_schema": RunAllWorkflowsInput,
        "boolean_args": ["strict", "clean_up_failure_management", "overwrite_existing"]
    },
]
