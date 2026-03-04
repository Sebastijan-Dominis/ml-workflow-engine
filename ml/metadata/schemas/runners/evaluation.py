"""Metadata models for evaluation runs."""
from typing import Literal

from ml.modeling.models.artifacts import Artifacts
from ml.modeling.models.config_fingerprint import ConfigFingerprint
from ml.modeling.models.experiment_lineage import ExperimentLineage
from ml.modeling.models.run_identity import RunIdentity
from ml.runners.evaluation.models.predictions import PredictionsPathsAndHashes
from pydantic import BaseModel


class EvaluationRunIdentity(RunIdentity):
    """
    Identity of a run, including stage, identifiers, and status.
    """
    stage: Literal["evaluation"] = "evaluation"
    eval_run_id: str

class EvaluationArtifacts(Artifacts, PredictionsPathsAndHashes):
    """
    Artifacts produced during evaluation, including paths and hashes.
    """
    metrics_path: str
    metrics_hash: str

class EvaluationMetadata(BaseModel):
    """
    Metadata for evaluation results.
    """
    run_identity: EvaluationRunIdentity
    lineage: ExperimentLineage
    config_fingerprint: ConfigFingerprint
    artifacts: EvaluationArtifacts
