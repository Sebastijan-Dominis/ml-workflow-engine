from typing import List, Dict, Optional, Union
from pydantic import BaseModel

class HardwareConfig(BaseModel):
    task_type: str
    devices: List[int]

class ParamDistributions(BaseModel):
    Model__depth: Optional[List[int]] = None
    Model__learning_rate: Optional[List[float]] = None
    Model__l2_leaf_reg: Optional[List[float]] = None
    Model__bagging_temperature: Optional[List[float]] = None
    Model__border_count: Optional[List[int]] = None
    Model__min_data_in_leaf: Optional[List[int]] = None
    Model__colsample_bylevel: Optional[List[float]] = None
    Model__random_strength: Optional[List[float]] = None

class BroadSearchConfig(BaseModel):
    iterations: int
    n_iter: int
    param_distributions: ParamDistributions

class NarrowIntParam(BaseModel):
    include: bool
    offsets: Optional[List[int]] = None
    low: Optional[int] = None
    high: Optional[int] = None

class NarrowFloatParam(BaseModel):
    include: bool
    factors: Optional[List[float]] = None
    low: Optional[float] = None
    high: Optional[float] = None

class NarrowSearchConfig(BaseModel):
    enabled: bool
    iterations: Optional[int] = None
    n_iter: Optional[int] = None
    Model__depth: Optional[NarrowIntParam] = None
    Model__learning_rate: Optional[NarrowFloatParam] = None
    Model__l2_leaf_reg: Optional[NarrowIntParam] = None
    Model__bagging_temperature: Optional[NarrowFloatParam] = None
    Model__min_data_in_leaf: Optional[NarrowIntParam] = None
    Model__random_strength: Optional[NarrowIntParam] = None
    Model__border_count: Optional[NarrowIntParam] = None
    Model__colsample_bylevel: Optional[NarrowFloatParam] = None

class SearchConfig(BaseModel):
    cv: int
    scoring: str
    random_state: int
    hardware: HardwareConfig
    seed: int
    verbose: Optional[int] = 100
    broad_search: BroadSearchConfig
    narrow_search: Optional[NarrowSearchConfig] = None