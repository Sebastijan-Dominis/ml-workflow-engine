# === Base parameter schema ===
from pydantic import BaseModel
from typing import Optional


class BaseModelParams(BaseModel):
    depth: Optional[int] = None
    learning_rate: Optional[float] = None
    l2_leaf_reg: Optional[float] = None
    random_strength: Optional[float] = None
    min_data_in_leaf: Optional[int] = None
    colsample_bylevel: Optional[float] = None
    border_count: Optional[int] = None

class BaseEnsembleParams(BaseModel):
    bagging_temperature: Optional[float] = None
