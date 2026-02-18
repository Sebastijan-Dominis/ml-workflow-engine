# Modularize if new datasets are added in the future, or if the processed config becomes too large. This will help keep the code organized and maintainable.

import logging

from pydantic import BaseModel, Field, model_validator

from ml.data.utils.config.schemas.shared import DatasetInfo, Input
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Create appropriate code logic before adding a new column here
SUPPORTED_COLUMNS = ["arrival_date"]

class ProcessedConfig(BaseModel):
    dataset: DatasetInfo
    input: Input
    remove_columns: list[str] = Field(default_factory=lambda: ["name", "email", "phone_number", "credit_card"], description="List of column names to remove from the dataset during processing.")
    create_columns: list[str] = Field(default_factory=list, description=f"List of new columns to create during processing. Supported columns are: {SUPPORTED_COLUMNS}")

    @model_validator(mode="after")
    def validate_create_columns(self: "ProcessedConfig") -> "ProcessedConfig":
        for new_col in self.create_columns:
            if new_col not in SUPPORTED_COLUMNS:
                msg = f"Unsupported column '{new_col}' in create_columns. Supported columns are: {SUPPORTED_COLUMNS}"
                logger.error(msg)
                raise ConfigError(msg)
        return self