from typing import Literal

from pydantic import BaseModel

from ml.config.schemas.hardware_cfg import HardwareConfig
from ml.config.schemas.model_specs import ClassWeightingConfig, TargetTransformConfig
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.types.splits import AllSplitsInfo


class Sources(BaseModel):
    """Represents the source information for the search experiment, including main source and any extended sources."""
    main: str
    extends: list[str] = []

class ExperimentMetadata(BaseModel):
    """Structured representation of experiment metadata, capturing configuration, environment, and lineage details."""
    problem : str
    segment: str
    version: str
    experiment_id: str
    sources: Sources
    env: Literal["default", "dev", "test", "prod"] = "default"
    best_params_path: str
    algorithm: str
    pipeline_version: str
    created_by: str
    created_at: str
    owner: str
    feature_lineage: list[FeatureLineage]
    seed: int
    hardware: HardwareConfig
    git_commit: str
    config_hash: str
    validation_status: str
    pipeline_hash: str
    scoring_method: str
    splits_info: AllSplitsInfo
    target_transform: TargetTransformConfig | None = None
    class_weighting: ClassWeightingConfig | None = None
