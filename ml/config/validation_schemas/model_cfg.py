"""Top-level validated schemas for search and train model configurations."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from ml.config.validation_schemas.model_specs import ModelSpecs
from ml.config.validation_schemas.search_cfg import SearchConfig
from ml.config.validation_schemas.train_cfg import TrainConfig


class SearchLineageConfig(BaseModel):
    """Lineage metadata captured when creating search configurations."""

    created_by: str
    created_at: datetime

class TrainingLineageConfig(BaseModel):
    """Lineage metadata captured when creating training configurations."""

    created_by: str
    created_at: datetime

class SearchModelConfig(ModelSpecs):
    """Validated model specification used by hyperparameter search stage."""

    extends: list[str] = Field(default_factory=list, description="List of config names to extend from, in order. Configs will be merged in the order they are listed, with later configs taking precedence in case of conflicts.")
    search: SearchConfig
    seed: int
    cv: int
    verbose: Optional[int] = 100
    search_lineage: SearchLineageConfig
    training: Optional[dict[str, Any]] = None  # inherited from shared defaults, unused

    class Config:
        """Pydantic options for strict schema validation behavior."""

        extra = "forbid"

class TrainModelConfig(ModelSpecs):
    """Validated model specification used by model training stage."""

    extends: list[str] = Field(default_factory=list, description="List of config names to extend from, in order. Configs will be merged in the order they are listed, with later configs taking precedence in case of conflicts.")
    training: TrainConfig
    seed: int
    cv: int
    verbose: Optional[int] = 100
    training_lineage: TrainingLineageConfig
    search: Optional[dict[str, Any]] = None  # inherited from shared defaults, unused

    class Config:
        """Pydantic options for strict schema validation behavior."""

        extra = "forbid"