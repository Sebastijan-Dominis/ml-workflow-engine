"""A module defining the data model for a registry entry in the ML promotion process. This includes details about the experiment, artifacts, feature lineage, and metrics associated with a promoted model."""
from typing import Literal

from pydantic import BaseModel


class RegistryArtifacts(BaseModel):
    model_hash: str
    model_path: str
    pipeline_hash: str | None = None
    pipeline_path: str | None = None
    top_k_feature_importances_path: str | None = None
    top_k_feature_importances_hash: str | None = None
    top_k_shap_importances_path: str | None = None
    top_k_shap_importances_hash: str | None = None

class RegistryFeatureSetLineage(BaseModel):
    name: str
    version: str
    snapshot_id: str
    file_hash: str
    in_memory_hash: str
    feature_schema_hash: str
    operator_hash: str
    feature_type: Literal["tabular"]

class RegistryEntryMetrics(BaseModel):
    train: dict[str, float | int]
    val: dict[str, float | int]
    test: dict[str, float | int]

class RegistryEntry(BaseModel):
    experiment_id: str
    train_run_id: str
    eval_run_id: str
    explain_run_id: str
    model_version: str
    pipeline_cfg_hash: str | None = None
    artifacts: RegistryArtifacts
    feature_lineage: list[RegistryFeatureSetLineage]
    metrics: RegistryEntryMetrics
    git_commit: str
    promotion_id: str
    promoted_at: str
