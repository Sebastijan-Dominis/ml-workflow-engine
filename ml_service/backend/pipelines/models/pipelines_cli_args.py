"""A module containing the input models for the pipelines CLI."""
from typing import Literal

from pydantic import BaseModel

LOGGING_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class RegisterRawSnapshotInput(BaseModel):
    """Model for the input of the register_raw_snapshot pipeline."""
    data: str
    version: str
    snapshot_id: str | None = "latest"
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"

class BuildInterimDatasetInput(BaseModel):
    """Model for the input of the build_interim_dataset pipeline."""
    data: str
    version: str
    raw_snapshot_id: str | None = "latest"
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"

class BuildProcessedDatasetInput(BaseModel):
    """Model for the input of the build_processed_dataset pipeline."""
    data: str
    version: str
    interim_snapshot_id: str | None = "latest"
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"

class FreezeFeaturesInput(BaseModel):
    """Model for the input of the freeze_features pipeline."""
    feature_set: str
    version: str
    owner: str | None = "Sebastijan"
    logging_level: LOGGING_LEVEL = "INFO"

class SearchInput(BaseModel):
    """Model for the input of the search pipeline."""
    problem: str
    segment: str
    version: str
    experiment_id: str | None = None
    env: str | None = "default"
    strict: bool | None = True
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"
    clean_up_failure_management: bool | None = True
    overwrite_existing: bool | None = False

class TrainInput(BaseModel):
    """Model for the input of the train pipeline."""
    problem: str
    segment: str
    version: str
    train_run_id: str | None = None
    experiment_id: str | None = None
    env: str | None = "default"
    strict: bool | None = True
    logging_level: LOGGING_LEVEL = "INFO"
    clean_up_failure_management: bool | None = True
    overwrite_existing: bool | None = False

class EvaluateInput(BaseModel):
    """Model for the input of the evaluate pipeline."""
    problem: str
    segment: str
    version: str
    experiment_id: str | None = None
    train_id: str | None = None
    env: str | None = "default"
    strict: bool | None = True
    logging_level: LOGGING_LEVEL = "INFO"

class ExplainInput(BaseModel):
    """Model for the input of the explain pipeline."""
    problem: str
    segment: str
    version: str
    experiment_id: str | None = None
    train_id: str | None = None
    top_k: int | None = None
    env: str | None = "default"
    strict: bool | None = True
    logging_level: LOGGING_LEVEL = "INFO"

class PromoteInput(BaseModel):
    """Model for the input of the promote pipeline."""
    problem: str
    segment: str
    version: str
    experiment_id: str
    train_run_id: str
    eval_run_id: str
    explain_run_id: str
    stage: str
    logging_level: LOGGING_LEVEL = "INFO"

class ExecuteAllDataPreprocessingInput(BaseModel):
    """Model for the input of the execute_all_data_preprocessing pipeline."""
    skip_if_existing: bool = True

class FreezeAllFeatureSetsInput(BaseModel):
    """Model for the input of the freeze_all_feature_sets pipeline."""
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"
    skip_if_existing: bool = True

class ExecuteExperimentWithLatestInput(BaseModel):
    """Model for the input of the execute_experiment_with_latest_input pipeline."""
    problem: str
    segment: str
    env: str | None = "default"
    strict: bool | None = True
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"
    clean_up_failure_management: bool | None = True
    version: str
    experiment_id: str | None = None
    overwrite_existing: bool | None = False
    top_k: int | None = None

class ExecuteAllExperimentsWithLatestInput(BaseModel):
    """Model for the input of the execute_all_experiments_with_latest_input pipeline."""
    env: str | None = "dev"
    strict: bool | None = True
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"
    clean_up_failure_management: bool | None = True
    overwrite_existing: bool | None = False
    top_k: int | None = None
    skip_if_existing: bool = True

class RunAllWorkflowsInput(BaseModel):
    """Model for the input of the run_all_workflows pipeline."""
    env: str | None = "dev"
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"
    skip_if_existing: bool = True
