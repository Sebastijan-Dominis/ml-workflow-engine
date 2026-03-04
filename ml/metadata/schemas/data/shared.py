"""Schemas for shared metadata components used across different dataset types."""
from pydantic import BaseModel

from ml.data.config.schemas.shared import DataInfo


class DataBasic(BaseModel):
    """Schema for basic data information."""
    name: str
    version: str
    format: str

class SourceData(DataBasic):
    """Schema for source data information in the metadata."""
    snapshot_id: str
    path: str

class CurrentData(DataInfo):
    """Schema for current data information in the metadata."""
    hash: str

class DataMemory(BaseModel):
    """Schema for memory usage information in the metadata."""
    old_memory_mb: float
    new_memory_mb: float
    change_mb: float
    change_percentage: float

class Columns(BaseModel):
    """Schema for column information in the metadata."""
    count: int
    names: list[str]
    dtypes: dict[str, str]

class DataRuntimeInfo(BaseModel):
    """Schema for runtime information of the data processing."""
    pandas_version: str
    numpy_version: str
    yaml_version: str
    python_version: str

class MetadataBase(BaseModel):
    """Base schema for dataset metadata."""
    rows: int
    columns: Columns
    created_at: str
    created_by: str
    owner: str

class SharedInterimProcessedMetadata(MetadataBase):
    """Base schema for metadata of both interim and processed datasets."""
    source_data: SourceData
    data: CurrentData
    memory: DataMemory
    config_hash: str
    duration: float
    runtime_info: DataRuntimeInfo