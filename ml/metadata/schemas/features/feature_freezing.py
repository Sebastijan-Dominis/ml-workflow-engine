"""Schema for metadata of a frozen feature snapshot."""
from pydantic import BaseModel

from ml.feature_freezing.models.freeze_runtime import FreezeRuntimeInfo
from ml.types import DataLineageEntry


class FreezeMetadata(BaseModel):
    """Schema for metadata of a frozen feature snapshot."""
    created_by: str
    created_at: str
    owner: str
    feature_type: str
    snapshot_path: str
    snapshot_id: str
    schema_path: str
    data_lineage: list[DataLineageEntry]
    in_memory_hash: str
    file_hash: str
    operator_hash: str
    config_hash: str
    feature_schema_hash: str
    runtime: FreezeRuntimeInfo
    row_count: int
    column_count: int
    duration_seconds: float
