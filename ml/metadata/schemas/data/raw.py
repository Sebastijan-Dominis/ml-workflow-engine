"""Schemas for raw snapshot metadata."""
from ml.metadata.schemas.data.shared import Columns, DataBasic, MetadataBase


class RawData(DataBasic):
    """Schema for raw data information in the metadata."""
    path_suffix: str
    hash: str

class RawSnapshotMetadata(MetadataBase):
    """Schema for the metadata of a raw data snapshot."""
    data: RawData
    memory_usage_mb: float
    raw_run_id: str