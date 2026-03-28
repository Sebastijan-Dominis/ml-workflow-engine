"""Models for inference metadata."""
from typing import Literal

from pydantic import BaseModel

from ml.modeling.models.feature_lineage import FeatureLineage


class InferenceMetadata(BaseModel):
    problem_type: str
    segment: str
    model_version: str
    model_stage: str
    run_id: str
    timestamp: str
    columns: list[str]
    snapshot_bindings_id: str
    feature_lineage: list[FeatureLineage]
    artifact_type: Literal["pipeline", "model"]
    artifact_hash: str
    inference_latency_seconds: float
