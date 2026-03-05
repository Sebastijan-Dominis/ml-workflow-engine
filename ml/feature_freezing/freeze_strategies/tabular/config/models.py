"""Pydantic schemas for tabular feature-freezing strategy configuration."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from ml.exceptions import ConfigError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

class DatasetConfig(BaseModel):
    """Source dataset definition used for feature freezing ingestion."""

    ref: str = Field("data/processed", description="Reference path for the dataset, e.g., 'data/processed'")
    name: str = Field(..., description="Name of the dataset, e.g., 'hotel_bookings'")
    version: str = Field(..., description="Version of the dataset, e.g., 'v1'")
    format: Literal["csv", "parquet"]
    merge_key: str = Field("row_id", description="Key to merge datasets on, default is 'row_id'")
    path_suffix: str = Field("data.{format}", description="Suffix for the dataset file, supports {format} placeholder")

class FeatureRolesConfig(BaseModel):
    """Feature role partitioning for validation and downstream usage."""

    categorical: list[str]
    numerical: list[str]
    datetime: list[str]

class OperatorsConfig(BaseModel):
    """Operator execution settings and reproducibility hash metadata."""

    mode: Literal["materialized","logical"]
    names: list[str]
    hash: str
    required_features: dict[str, list[str]]

class ConstraintsConfig(BaseModel):
    """Data quality constraints enforced on frozen features."""

    forbid_nulls: list[str]
    max_cardinality: dict[str, int]

class StorageConfig(BaseModel):
    """Snapshot storage format and compression settings."""

    format: Literal["parquet"]
    compression: str | None = "snappy"

class LineageConfig(BaseModel):
    """Lineage metadata for feature registry config provenance."""

    created_by: str
    created_at: datetime

class TabularFeaturesConfig(BaseModel):
    """Top-level validated config for tabular feature freezing."""

    type: str = "tabular"
    description: str | None = None
    data: list[DatasetConfig]
    min_rows: int = Field(default=1000, ge=0)
    feature_store_path: Path
    columns: list[str]
    feature_roles: FeatureRolesConfig
    operators: OperatorsConfig | None = None
    constraints: ConstraintsConfig
    storage: StorageConfig
    lineage: LineageConfig
    ...
    model_config = ConfigDict(extra="forbid")  # Pydantic options for strict schema validation behavior

    @field_validator("operators", mode="before")
    def required_features_must_equal_operator_names(cls, v):
        """Validate operator names align with required-features mapping keys.

        Args:
            v: Raw operators payload.

        Returns:
            Any: Validated operators payload.
        """

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
    def validate_feature_roles_match_columns(self):
        """Ensure feature role assignments exactly match included columns.

        Args:
            self: Candidate tabular features config.

        Returns:
            TabularFeaturesConfig: Validated config.
        """
        columns = set(self.columns)
        roles = set(self.feature_roles.categorical + self.feature_roles.numerical + self.feature_roles.datetime)
        if columns != roles:
            missing_in_roles = columns - roles
            extra_in_roles = roles - columns
            msg = (
                f"Feature roles do not match included columns. "
                f"Missing in roles: {missing_in_roles}, extra in roles: {extra_in_roles}."
            )
            logger.error(msg)
            raise ConfigError(msg)
        return self

    @model_validator(mode="after")
    def validate_constraints_match_columns(self):
        """Ensure constraint-referenced columns exist in included columns.

        Args:
            self: Candidate tabular features config.

        Returns:
            TabularFeaturesConfig: Validated config."""
        columns = set(self.columns)
        forbidden_nulls = set(self.constraints.forbid_nulls)
        max_cardinality_cols = set(self.constraints.max_cardinality.keys())

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

        return self

    @model_validator(mode="after")
    def validate_required_features_for_operators(self):
        """Ensure all operator required features are present in included columns.

        Args:
            self: Candidate tabular features config.

        Returns:
            TabularFeaturesConfig: Validated config.
        """
        if self.operators is None:
            return self

        columns = set(self.columns)
        for op_name, req_feats in self.operators.required_features.items():
            missing = set(req_feats) - columns
            if missing:
                msg = (
                    f"Required features {missing} for operator {op_name} "
                    f"are not in included columns {columns}"
                )
                logger.error(msg)
                raise ConfigError(msg)

        return self
