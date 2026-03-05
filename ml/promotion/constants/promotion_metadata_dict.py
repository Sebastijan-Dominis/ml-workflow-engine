from typing import TypedDict


class RunIdentity(TypedDict, total=False):
    experiment_id: str | None
    train_run_id: str | None
    eval_run_id: str | None
    explain_run_id: str | None
    stage: str
    promotion_id: str | None
    staging_id: str | None

class Decision(TypedDict, total=False):
    promoted: bool
    reason: str
    beats_previous: bool | None

class Context(TypedDict):
    git_commit: str
    promotion_conda_env_hash: str
    training_conda_env_hash: str
    timestamp: str

class PromotionMetadataDict(TypedDict):
    run_identity: RunIdentity
    previous_production_run_identity: RunIdentity
    metrics: dict
    previous_production_metrics: dict | None
    promotion_thresholds: dict
    promotion_thresholds_hash: str
    decision: Decision
    context: Context
