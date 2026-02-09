# === Base parameter schema ===
from typing import Optional

from pydantic import BaseModel


class BaseModelParams(BaseModel):
    depth: Optional[int] = None
    learning_rate: Optional[float] = None
    l2_leaf_reg: Optional[float] = None
    random_strength: Optional[float] = None
    min_data_in_leaf: Optional[int] = None
    border_count: Optional[int] = None

class BaseEnsembleParams(BaseModel):
    bagging_temperature: Optional[float] = None
    colsample_bylevel: Optional[float] = None
