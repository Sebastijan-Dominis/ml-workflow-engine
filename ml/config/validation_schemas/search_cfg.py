from typing import Optional

from pydantic import BaseModel, Field

from ml.config.validation_schemas.hardware_cfg import HardwareConfig


# === Broad search model/ensemble params ===
class BroadModelParams(BaseModel):
    depth: Optional[list[int]] = None
    learning_rate: Optional[list[float]] = None
    l2_leaf_reg: Optional[list[float]] = None
    colsample_bylevel: Optional[list[float]] = None
    random_strength: Optional[list[float]] = None
    min_data_in_leaf: Optional[list[int]] = None
    border_count: Optional[list[int]] = None

class BroadEnsembleParams(BaseModel):
    bagging_temperature: Optional[list[float]] = None

class BroadParamDistributions(BaseModel):
    model: BroadModelParams = Field(default_factory=BroadModelParams)
    ensemble: BroadEnsembleParams = Field(default_factory=BroadEnsembleParams)

    def to_flat_dict(self, prefix_map: dict | None = None) -> dict:
        """
        Flatten structured params for grid/random search.
        All keys use 'Model__' prefix for CatBoost compatibility.
        """
        prefix_map = prefix_map or {"model": "Model", "ensemble": "Model"}
        flat = {}
        for key, sub in self.model_dump(exclude_none=True).items():
            prefix = prefix_map.get(key, key)
            for subkey, val in sub.items():
                flat[f"{prefix}__{subkey}"] = val
        return flat

# === Narrow search schemas ===
class NarrowIntParam(BaseModel):
    include: bool
    offsets: Optional[list[int]] = None
    low: Optional[int] = None
    high: Optional[int] = None

class NarrowFloatParam(BaseModel):
    include: bool
    factors: Optional[list[float]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    decimals: Optional[int] = None

class NarrowModelParams(BaseModel):
    depth: Optional[NarrowIntParam] = None
    learning_rate: Optional[NarrowFloatParam] = None
    l2_leaf_reg: Optional[NarrowFloatParam] = None
    colsample_bylevel: Optional[NarrowFloatParam] = None
    random_strength: Optional[NarrowFloatParam] = None
    min_data_in_leaf: Optional[NarrowIntParam] = None
    border_count: Optional[NarrowIntParam] = None

class NarrowEnsembleParams(BaseModel):
    bagging_temperature: Optional[NarrowFloatParam] = None

class NarrowParamConfig(BaseModel):
    model: NarrowModelParams = Field(default_factory=NarrowModelParams)
    ensemble: NarrowEnsembleParams = Field(default_factory=NarrowEnsembleParams)

class NarrowSearchConfig(BaseModel):
    enabled: bool
    iterations: int
    n_iter: int
    param_configurations: NarrowParamConfig = Field(default_factory=NarrowParamConfig)

# === Broad search ===
class BroadSearchConfig(BaseModel):
    iterations: int
    n_iter: int
    param_distributions: BroadParamDistributions = Field(default_factory=BroadParamDistributions)

# === Full Search Config ===
class SearchConfig(BaseModel):
    random_state: int
    broad: BroadSearchConfig
    narrow: NarrowSearchConfig = Field(default_factory=lambda: NarrowSearchConfig(
        enabled=False,
        iterations=0,
        n_iter=0,
    ))
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    error_score: Optional[str] = None
