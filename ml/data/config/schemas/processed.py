"""Validation schemas for processed-stage data pipeline configuration."""

import logging
from datetime import datetime

from ml.data.config.schemas.shared import DataInfo
from ml.exceptions import ConfigError
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

class LineageConfig(BaseModel):
    """Lineage metadata describing processed config provenance."""

    created_by: str
    created_at: datetime

class ProcessedConfig(BaseModel):
    """Top-level validated configuration for processed data generation."""

    data: DataInfo
    interim_data_version: str
    remove_columns: list[str] = Field(default_factory=lambda: ["name", "email", "phone_number", "credit_card"], description="List of column names to remove from the data during processing.")
    lineage: LineageConfig

    # ensure that the interim_data_version is not empty and follows a specific format (e.g., v1, v2, etc.)
    @model_validator(mode="after")
    def validate_interim_data_version(cls, config):
        """Validate that ``interim_data_version`` follows the ``v{number}`` format.

        Args:
            cls: Pydantic model class invoking the validator.
            config: Parsed ``ProcessedConfig`` instance being validated.

        Returns:
            The validated ``ProcessedConfig`` instance.
        """
        if not config.interim_data_version.startswith("v") or not config.interim_data_version[1:].isdigit():
            msg = f"Invalid interim_data_version '{config.interim_data_version}'. It must start with 'v' followed by a number (e.g., v1, v2)."
            logger.error(msg)
            raise ConfigError(msg)
        return config
