from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from ml.config.validation_schemas.model_specs import ModelSpecs
from ml.config.validation_schemas.search_cfg import SearchConfig
from ml.config.validation_schemas.train_cfg import TrainConfig


class SearchLineageConfig(BaseModel):
    created_by: str
    created_at: datetime

class TrainingLineageConfig(BaseModel):
    created_by: str
    created_at: datetime

class SearchModelConfig(ModelSpecs):
    extends: list[str] = Field(default_factory=list, description="List of config names to extend from, in order. Configs will be merged in the order they are listed, with later configs taking precedence in case of conflicts.")
    search: SearchConfig
    seed: int
    cv: int
    verbose: Optional[int] = 100
    search_lineage: SearchLineageConfig
    training: Optional[dict[str, Any]] = None  # inherited from shared defaults, unused

    class Config:
        extra = "forbid"

class TrainModelConfig(ModelSpecs):
    extends: list[str] = Field(default_factory=list, description="List of config names to extend from, in order. Configs will be merged in the order they are listed, with later configs taking precedence in case of conflicts.")
    training: TrainConfig
    seed: int
    cv: int
    verbose: Optional[int] = 100
    training_lineage: TrainingLineageConfig
    search: Optional[dict[str, Any]] = None  # inherited from shared defaults, unused

    class Config:
        extra = "forbid"