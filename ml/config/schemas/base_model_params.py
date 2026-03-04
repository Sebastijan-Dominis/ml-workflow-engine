"""Shared base parameter schemas for train/search model configurations."""


from pydantic import BaseModel


class BaseModelParams(BaseModel):
    """Core model hyperparameters shared across workflows."""

    depth: int | None = None
    learning_rate: float | None = None
    l2_leaf_reg: float | None = None
    random_strength: float | None = None
    min_data_in_leaf: int | None = None
    border_count: int | None = None

class BaseEnsembleParams(BaseModel):
    """Ensemble-related hyperparameters shared across workflows."""

    bagging_temperature: float | None = None
    colsample_bylevel: float | None = None
