"""Validation schemas for broad/narrow hyperparameter search settings."""


from ml.config.schemas.hardware_cfg import HardwareConfig, HardwareTaskType
from pydantic import BaseModel, Field


# === Broad search model/ensemble params ===
class BroadModelParams(BaseModel):
    """Broad-search parameter candidates for model-level hyperparameters."""

    depth: list[int] | None = None
    learning_rate: list[float] | None = None
    l2_leaf_reg: list[float] | None = None
    colsample_bylevel: list[float] | None = None
    random_strength: list[float] | None = None
    min_data_in_leaf: list[int] | None = None
    border_count: list[int] | None = None

class BroadEnsembleParams(BaseModel):
    """Broad-search parameter candidates for ensemble-level hyperparameters."""

    bagging_temperature: list[float] | None = None

class BroadParamDistributions(BaseModel):
    """Container for broad-search parameter distributions."""

    model: BroadModelParams = Field(default_factory=BroadModelParams)
    ensemble: BroadEnsembleParams = Field(default_factory=BroadEnsembleParams)

    def to_flat_dict(self, prefix_map: dict | None = None) -> dict:
        """
        Flatten structured params for grid/random search.
        All keys use 'Model__' prefix for CatBoost compatibility.

        Args:
            prefix_map: Optional mapping from top-level blocks (for example,
                ``model`` and ``ensemble``) to flattened key prefixes.

        Returns:
            Flattened parameter dictionary suitable for search estimators.
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
    """Narrow-search configuration for integer-valued parameters."""

    include: bool
    offsets: list[int] | None = None
    low: int | None = None
    high: int | None = None

class NarrowFloatParam(BaseModel):
    """Narrow-search configuration for float-valued parameters."""

    include: bool
    factors: list[float] | None = None
    low: float | None = None
    high: float | None = None
    decimals: int | None = None

class NarrowModelParams(BaseModel):
    """Narrow-search parameter rules for model-level hyperparameters."""

    depth: NarrowIntParam | None = None
    learning_rate: NarrowFloatParam | None = None
    l2_leaf_reg: NarrowFloatParam | None = None
    colsample_bylevel: NarrowFloatParam | None = None
    random_strength: NarrowFloatParam | None = None
    min_data_in_leaf: NarrowIntParam | None = None
    border_count: NarrowIntParam | None = None

class NarrowEnsembleParams(BaseModel):
    """Narrow-search parameter rules for ensemble hyperparameters."""

    bagging_temperature: NarrowFloatParam | None = None

class NarrowParamConfig(BaseModel):
    """Container for narrow-search parameter configuration blocks."""

    model: NarrowModelParams = Field(default_factory=NarrowModelParams)
    ensemble: NarrowEnsembleParams = Field(default_factory=NarrowEnsembleParams)

class NarrowSearchConfig(BaseModel):
    """Configuration for refinement search around broad-search results."""

    enabled: bool
    iterations: int
    n_iter: int
    param_configurations: NarrowParamConfig = Field(default_factory=NarrowParamConfig)

# === Broad search ===
class BroadSearchConfig(BaseModel):
    """Configuration for broad hyperparameter search stage."""

    iterations: int
    n_iter: int
    param_distributions: BroadParamDistributions = Field(default_factory=BroadParamDistributions)

# === Full Search Config ===
class SearchConfig(BaseModel):
    """Full search-stage configuration including hardware and error policy."""

    random_state: int
    broad: BroadSearchConfig
    narrow: NarrowSearchConfig = Field(default_factory=lambda: NarrowSearchConfig(
        enabled=False,
        iterations=0,
        n_iter=0,
    ))
    hardware: HardwareConfig = Field(default_factory=lambda: HardwareConfig(task_type=HardwareTaskType.GPU))
    error_score: str | None = None
