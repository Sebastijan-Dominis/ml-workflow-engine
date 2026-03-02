"""Validation schemas for broad/narrow hyperparameter search settings."""

from typing import Optional

from pydantic import BaseModel, Field

from ml.config.validation_schemas.hardware_cfg import HardwareConfig


# === Broad search model/ensemble params ===
class BroadModelParams(BaseModel):
    """Broad-search parameter candidates for model-level hyperparameters."""

    depth: Optional[list[int]] = None
    learning_rate: Optional[list[float]] = None
    l2_leaf_reg: Optional[list[float]] = None
    colsample_bylevel: Optional[list[float]] = None
    random_strength: Optional[list[float]] = None
    min_data_in_leaf: Optional[list[int]] = None
    border_count: Optional[list[int]] = None

class BroadEnsembleParams(BaseModel):
    """Broad-search parameter candidates for ensemble-level hyperparameters."""

    bagging_temperature: Optional[list[float]] = None

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
    offsets: Optional[list[int]] = None
    low: Optional[int] = None
    high: Optional[int] = None

class NarrowFloatParam(BaseModel):
    """Narrow-search configuration for float-valued parameters."""

    include: bool
    factors: Optional[list[float]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    decimals: Optional[int] = None

class NarrowModelParams(BaseModel):
    """Narrow-search parameter rules for model-level hyperparameters."""

    depth: Optional[NarrowIntParam] = None
    learning_rate: Optional[NarrowFloatParam] = None
    l2_leaf_reg: Optional[NarrowFloatParam] = None
    colsample_bylevel: Optional[NarrowFloatParam] = None
    random_strength: Optional[NarrowFloatParam] = None
    min_data_in_leaf: Optional[NarrowIntParam] = None
    border_count: Optional[NarrowIntParam] = None

class NarrowEnsembleParams(BaseModel):
    """Narrow-search parameter rules for ensemble hyperparameters."""

    bagging_temperature: Optional[NarrowFloatParam] = None

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
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    error_score: Optional[str] = None
