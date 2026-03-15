from typing import Literal

from pydantic import BaseModel

LOGGING_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class RegisterRawSnapshotInput(BaseModel):
    data: str
    version: str
    snapshot_id: str | None = "latest"
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"

class BuildInterimDatasetInput(BaseModel):
    data: str
    version: str
    raw_snapshot_id: str | None = "latest"
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"

class BuildProcessedDatasetInput(BaseModel):
    data: str
    version: str
    interim_snapshot_id: str | None = "latest"
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"

class FreezeFeaturesInput(BaseModel):
    feature_set: str
    version: str
    owner: str | None = "Sebastijan"
    logging_level: LOGGING_LEVEL = "INFO"

class SearchInput(BaseModel):
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
    problem: str
    segment: str
    version: str
    experiment_id: str | None = None
    train_id: str | None = None
    env: str | None = "default"
    strict: bool | None = True
    logging_level: LOGGING_LEVEL = "INFO"

class ExplainInput(BaseModel):
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
    skip_if_existing: bool = True

class FreezeAllFeatureSetsInput(BaseModel):
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"
    skip_if_existing: bool = True

class ExecuteExperimentWithLatestInput(BaseModel):
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
    env: str | None = "dev"
    strict: bool | None = True
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"
    clean_up_failure_management: bool | None = True
    overwrite_existing: bool | None = False
    top_k: int | None = None
    skip_if_existing: bool = True

class RunAllWorkflowsInput(BaseModel):
    env: str | None = "dev"
    logging_level: LOGGING_LEVEL = "INFO"
    owner: str | None = "Sebastijan"
    skip_if_existing: bool = True
