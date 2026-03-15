from ml_service.backend.pipelines.models.pipelines_cli_args import (
    BuildInterimDatasetInput,
    BuildProcessedDatasetInput,
    EvaluateInput,
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

FRONTEND_PIPELINES_REGISTRY = [
    {
        "name": "Register Raw Snapshot",
        "endpoint": "pipelines/register_raw_snapshot",
        "args_schema": RegisterRawSnapshotInput,
    },
    {
        "name": "Build Interim Dataset",
        "endpoint": "pipelines/build_interim_dataset",
        "args_schema": BuildInterimDatasetInput,
    },
    {
        "name": "Build Processed Dataset",
        "endpoint": "pipelines/build_processed_dataset",
        "args_schema": BuildProcessedDatasetInput,
    },
    {
        "name": "Freeze Feature Set",
        "endpoint": "pipelines/freeze_feature_set",
        "args_schema": FreezeFeaturesInput,
    },
    {
        "name": "Search",
        "endpoint": "pipelines/search",
        "args_schema": SearchInput
    },
    {
        "name": "Train",
        "endpoint": "pipelines/train",
        "args_schema": TrainInput
    },
    {
        "name": "Evaluate",
        "endpoint": "pipelines/evaluate",
        "args_schema": EvaluateInput
    },
    {
        "name": "Explain",
        "endpoint": "pipelines/explain",
        "args_schema": ExplainInput
    },
    {
        "name": "Promote",
        "endpoint": "pipelines/promote",
        "args_schema": PromoteInput
    },
    {
        "name": "Freeze All Feature Sets",
        "endpoint": "pipelines/freeze_all_feature_sets",
        "args_schema": FreezeAllFeatureSetsInput
    },
    {
        "name": "Execute Experiment With Latest",
        "endpoint": "pipelines/execute_experiment_with_latest",
        "args_schema": ExecuteExperimentWithLatestInput
    },
    {
        "name": "Execute All Experiments With Latest",
        "endpoint": "pipelines/execute_all_experiments_with_latest",
        "args_schema": ExecuteAllExperimentsWithLatestInput
    },
    {
        "name": "Run All Workflows",
        "endpoint": "pipelines/run_all_workflows",
        "args_schema": RunAllWorkflowsInput
    }
]
