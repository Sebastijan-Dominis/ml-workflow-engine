# Modularize if new datas are added in the future, or if the processed config becomes too large. This will help keep the code organized and maintainable.

import logging
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from ml.data.utils.config.schemas.shared import DataInfo
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

class LineageConfig(BaseModel):
    created_by: str
    created_at: datetime

class ProcessedConfig(BaseModel):
    data: DataInfo
    remove_columns: list[str] = Field(default_factory=lambda: ["name", "email", "phone_number", "credit_card"], description="List of column names to remove from the data during processing.")
    lineage: LineageConfig