"""Module containing pydantic models for validating the modeling configuration section of the ML service."""
from dataclasses import dataclass

from ml.config.schemas.model_specs import ModelSpecs
from ml.config.schemas.search_cfg import SearchConfig
from ml.config.schemas.train_cfg import TrainConfig
from pydantic import BaseModel, Field


class SearchConfigForValidation(BaseModel):
    """Separate schema for search config validation to allow extra fields for lineage tracking and extension."""
    extends: list[str] = Field(default_factory=list)
    search_lineage: dict
    search: SearchConfig

class TrainConfigForValidation(BaseModel):
    """Separate schema for train config validation to allow extra fields for lineage tracking and extension."""
    extends: list[str] = Field(default_factory=list)
    training_lineage: dict
    training: TrainConfig

@dataclass
class RawConfigsWithLineage:
    """Dataclass to hold raw configs along with their lineage information."""
    model_specs: dict
    search: dict
    training: dict

@dataclass
class ValidatedConfigs:
    """Dataclass to hold validated configs along with their lineage information."""
    model_specs: ModelSpecs
    search: SearchConfigForValidation
    training: TrainConfigForValidation

@dataclass
class ConfigPaths:
    """Dataclass to hold the paths for each configuration file."""
    model_specs: str
    search: str
    training: str
