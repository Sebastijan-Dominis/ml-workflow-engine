"""Typed output contract for training runner implementations."""

from dataclasses import dataclass

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline
from typing import Optional

SUPPORTED_MODELS = CatBoostClassifier | CatBoostRegressor

@dataclass
class TRAIN_OUTPUT:
    """Container for trained artifacts, lineage, metrics, and pipeline hash."""

    model: SUPPORTED_MODELS
    pipeline: Optional[Pipeline] # Some trainers might return a pipeline, while others might not. This allows for flexibility in the return type.
    lineage: list[dict]
    metrics: dict[str, float]
    pipeline_cfg_hash: Optional[str] # Not all trainers might use a pipeline config, so this can be optional.
