"""Typed output contract for training runner implementations."""

from dataclasses import dataclass
from typing import TypeAlias

from catboost import CatBoostClassifier, CatBoostRegressor
from ml.modeling.models.feature_lineage import FeatureLineage
from sklearn.pipeline import Pipeline

SUPPORTED_MODELS: TypeAlias = CatBoostClassifier | CatBoostRegressor

@dataclass
class TrainOutput:
    """Container for trained artifacts, lineage, metrics, and pipeline hash."""

    model: SUPPORTED_MODELS
    pipeline: Pipeline | None # Some trainers might return a pipeline, while others might not. This allows for flexibility in the return type.
    lineage: list[FeatureLineage]
    metrics: dict[str, float]
    pipeline_cfg_hash: str | None # Not all trainers might use a pipeline config, so this can be optional.
