"""Models for promotion metadata."""
from typing import Literal

from pydantic import BaseModel

from ml.modeling.models.metrics import EvaluationMetricsHelper
from ml.promotion.config.models import PromotionThresholds


class RunIdentity(BaseModel):
    experiment_id: str
    train_run_id: str
    eval_run_id: str
    explain_run_id: str

class CurrentRunIdentity(RunIdentity):
    """Base model for current run identity, extended by staging and production identities."""
    stage: Literal["staging", "production"]

class CurrentProductionRunIdentity(CurrentRunIdentity):
    """Model for current production run identity, extending base run identity."""
    promotion_id: str

class CurrentStagingRunIdentity(CurrentRunIdentity):
    """Model for current staging run identity, extending base run identity."""
    staging_id: str

class PreviousProductionRunIdentity(BaseModel):
    """Model for previous production run identity, extending base run identity."""
    experiment_id: str | None = None
    train_run_id: str | None = None
    eval_run_id: str | None = None
    explain_run_id: str | None = None
    promotion_id: str | None = None  # promotion_id may be None if no previous production run exists

class PromotionDecision(BaseModel):
    """Base model for promotion decision results."""
    promoted: bool
    reason: str

class ProductionPromotionDecision(PromotionDecision):
    """Model for production promotion decision, extending base promotion decision."""
    beats_previous: bool

class Context(BaseModel):
    """Model for promotion execution context."""
    git_commit: str
    promotion_conda_env_hash: str
    training_conda_env_hash: str
    timestamp: str

class PromotionMetadata(BaseModel):
    """Base model for promotion metadata, containing all relevant information for decision making and metadata preparation."""
    previous_production_run_identity: PreviousProductionRunIdentity
    metrics: EvaluationMetricsHelper
    previous_production_metrics: EvaluationMetricsHelper | None = None
    promotion_thresholds: PromotionThresholds
    promotion_thresholds_hash: str
    context: Context

class ProductionPromotionMetadata(PromotionMetadata):
    """Model for production promotion metadata, extending base promotion metadata."""
    run_identity: CurrentProductionRunIdentity
    decision: ProductionPromotionDecision

class StagingPromotionMetadata(PromotionMetadata):
    """Model for staging promotion metadata, extending base promotion metadata."""
    run_identity: CurrentStagingRunIdentity
    decision: PromotionDecision
