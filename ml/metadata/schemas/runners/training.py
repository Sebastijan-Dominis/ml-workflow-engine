from typing import Literal

from ml.modeling.models.artifacts import Artifacts
from ml.modeling.models.config_fingerprint import ConfigFingerprint
from ml.modeling.models.experiment_lineage import ExperimentLineage
from ml.modeling.models.run_identity import RunIdentity
from pydantic import BaseModel


class TrainingRunIdentity(RunIdentity):
    stage: Literal["training"] = "training"

class TrainingMetadata(BaseModel):
    """Metadata for training a machine learning model."""
    run_identity: TrainingRunIdentity
    lineage: ExperimentLineage
    config_fingerprint: ConfigFingerprint
    artifacts: Artifacts
