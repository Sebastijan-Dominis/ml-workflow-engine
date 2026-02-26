import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

class DatasetConfig(BaseModel):
    ref: str = Field("data/processed", description="Reference path for the dataset, e.g., 'data/processed'")
    name: str = Field(..., description="Name of the dataset, e.g., 'hotel_bookings'")
    version: str = Field(..., description="Version of the dataset, e.g., 'v1'")
    format: Literal["csv", "parquet"]
    merge_key: str = Field("row_id", description="Key to merge datasets on, default is 'row_id'")
    path_suffix: str = Field("data.{format}", description="Suffix for the dataset file, supports {format} placeholder")

class FeatureRolesConfig(BaseModel):
    categorical: list[str]
    numerical: list[str]
    datetime: list[str]

class OperatorsConfig(BaseModel):
    mode: Literal["materialized","logical"]
    names: list[str]
    hash: str
    required_features: dict[str, list[str]]

class ConstraintsConfig(BaseModel):
    forbid_nulls: list[str]
    max_cardinality: dict[str, int]

class StorageConfig(BaseModel):
    format: Literal["parquet"]
    compression: Optional[str] = "snappy"

class LineageConfig(BaseModel):
    feature_set_version: str
    created_by: str
    created_at: str

class TabularFeaturesConfig(BaseModel):
    type: str = "tabular"
    description: str | None = None
    data: list[DatasetConfig]
    min_rows: int = Field(default=1000, ge=0)
    feature_store_path: Path
    columns: list[str]
    feature_roles: FeatureRolesConfig
    operators: Optional[OperatorsConfig] = None
    constraints: ConstraintsConfig
    storage: StorageConfig
    lineage: LineageConfig
    ...
    class Config:
        extra = "forbid"

    @field_validator("operators", mode="before")
    def required_features_must_equal_operator_names(cls, v):
        if v is None:
            return v
        names = set(v["names"])
        required = set(v["required_features"].keys())
        if names != required:
            msg = f"Operator names {names} must match required features {required}"
            logger.error(msg)
            raise ConfigError(msg)
        return v
    
    @model_validator(mode="after")
    def validate_feature_roles_match_columns(cls, config):
        columns = set(config.columns)
        roles = set(config.feature_roles.categorical + config.feature_roles.numerical + config.feature_roles.datetime)
        if columns != roles:
            missing_in_roles = columns - roles
            extra_in_roles = roles - columns
            msg = f"Feature roles do not match included columns. Missing in roles: {missing_in_roles}, extra in roles: {extra_in_roles}."
            logger.error(msg)
            raise ConfigError(msg)
        return config
    
    @model_validator(mode="after")
    def validate_constraints_match_columns(cls, config):
        columns = set(config.columns)
        forbidden_nulls = set(config.constraints.forbid_nulls)
        max_cardinality_cols = set(config.constraints.max_cardinality.keys())
        if not forbidden_nulls.issubset(columns):
            missing = forbidden_nulls - columns
            msg = f"Forbidden nulls {missing} are not in included columns {columns}"
            logger.error(msg)
            raise ConfigError(msg)
        if not max_cardinality_cols.issubset(columns):
            missing = max_cardinality_cols - columns
            msg = f"Max cardinality columns {missing} are not in included columns {columns}"
            logger.error(msg)
            raise ConfigError(msg)
        return config
    
    # validate that all of the required features for operators are included in columns
    @model_validator(mode="after")
    def validate_required_features_for_operators(cls, config):
        if config.operators is None:
            return config
        columns = set(config.columns)
        for op_name, req_feats in config.operators.required_features.items():
            missing = set(req_feats) - columns
            if missing:
                msg = f"Required features {missing} for operator {op_name} are not in included columns {columns}"
                logger.error(msg)
                raise ConfigError(msg)
        return config