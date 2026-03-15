from typing import Annotated

from fastapi import APIRouter, Body, Request
from ml_service.backend.main import limiter
from ml_service.backend.pipelines.execute_pipeline import execute_pipeline
from ml_service.backend.pipelines.models.pipelines_cli_args import (
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

router = APIRouter(prefix="/pipelines", tags=["pipelines"])

@router.post("/register_raw_snapshot", status_code=200)
@limiter.limit("1/30seconds")
def register_raw_snapshot(payload: Annotated[RegisterRawSnapshotInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.data.register_raw_snapshot",
        payload=payload,
        boolean_args=[],
    )

@router.post("/build_interim_dataset", status_code=200)
@limiter.limit("1/30seconds")
def build_interim_dataset(payload: Annotated[BuildInterimDatasetInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.data.build_interim_dataset",
        payload=payload,
        boolean_args=[],
    )

@router.post("/build_processed_dataset", status_code=200)
@limiter.limit("1/30seconds")
def build_processed_dataset(payload: Annotated[BuildProcessedDatasetInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.data.build_processed_dataset",
        payload=payload,
        boolean_args=[],
    )

@router.post("/freeze_feature_set", status_code=200)
@limiter.limit("1/15seconds")
def freeze_feature_set(payload: Annotated[FreezeFeaturesInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.features.freeze",
        payload=payload,
        boolean_args=[],
    )

@router.post("/search", status_code=200)
@limiter.limit("1/minute")
def search(payload: Annotated[SearchInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.search.search",
        payload=payload,
        boolean_args=["strict", "clean_up_failure_management", "overwrite_existing"],
    )

@router.post("/train", status_code=200)
@limiter.limit("1/minute")
def train(payload: Annotated[TrainInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.runners.training.train",
        payload=payload,
        boolean_args=["strict", "clean_up_failure_management", "overwrite_existing"],
    )

@router.post("/evaluate", status_code=200)
@limiter.limit("1/30seconds")
def evaluate(payload: Annotated[EvaluateInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.runners.evaluation.evaluate",
        payload=payload,
        boolean_args=["strict", "clean_up_failure_management", "overwrite_existing"],
    )

@router.post("/explain", status_code=200)
@limiter.limit("1/30seconds")
def explain(payload: Annotated[ExplainInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.runners.explainability.explain",
        payload=payload,
        boolean_args=["strict", "clean_up_failure_management", "overwrite_existing"],
    )

@router.post("/promote", status_code=200)
@limiter.limit("1/minute")
def promote(payload: Annotated[PromoteInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.promotion.promote",
        payload=payload,
        boolean_args=[],
    )

@router.post("/execute_all_data_preprocessing", status_code=200)
@limiter.limit("1/10minutes; 3/hour")
def execute_all_data_preprocessing(payload: Annotated[ExecuteAllDataPreprocessingInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.orchestration.data.execute_all_data_preprocessing",
        payload=payload,
        boolean_args=[],
    )

@router.post("/freeze_all_feature_sets", status_code=200)
@limiter.limit("1/10minutes")
def freeze_all_feature_sets(payload: Annotated[FreezeAllFeatureSetsInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.orchestration.features.freeze_all_feature_sets",
        payload=payload,
        boolean_args=[],
    )

@router.post("/execute_experiment_with_latest", status_code=200)
@limiter.limit("1/10minutes; 3/hour")
def execute_experiment_with_latest(payload: Annotated[ExecuteExperimentWithLatestInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.orchestration.experiments.execute_experiment_with_latest",
        payload=payload,
        boolean_args=["strict", "clean_up_failure_management", "overwrite_existing"],
    )

@router.post("/execute_all_experiments_with_latest", status_code=200)
@limiter.limit("1/hour; 5/day")
def execute_all_experiments_with_latest(payload: Annotated[ExecuteAllExperimentsWithLatestInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.orchestration.experiments.execute_all_experiments_with_latest",
        payload=payload,
        boolean_args=[],
    )

@router.post("/run_all_workflows", status_code=200)
@limiter.limit("1/hour; 5/day")
def run_all_workflows(payload: Annotated[RunAllWorkflowsInput, Body(...)], request: Request): # type: ignore
    return execute_pipeline(
        module_path="pipelines.orchestration.run_all_workflows",
        payload=payload,
        boolean_args=[],
    )
