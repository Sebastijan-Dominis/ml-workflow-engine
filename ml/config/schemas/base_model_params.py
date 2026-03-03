"""Shared base parameter schemas for train/search model configurations."""

from typing import Optional

from pydantic import BaseModel


class BaseModelParams(BaseModel):
    """Core model hyperparameters shared across workflows."""

    depth: Optional[int] = None
    learning_rate: Optional[float] = None
    l2_leaf_reg: Optional[float] = None
    random_strength: Optional[float] = None
    min_data_in_leaf: Optional[int] = None
    border_count: Optional[int] = None

class BaseEnsembleParams(BaseModel):
    """Ensemble-related hyperparameters shared across workflows."""

    bagging_temperature: Optional[float] = None
    colsample_bylevel: Optional[float] = None
