"""Schemas for processed dataset metadata."""
from ml.metadata.schemas.data.shared import SharedInterimProcessedMetadata
from pydantic import BaseModel


class RowIdInfo(BaseModel):
    """Schema for information about row identifiers in the processed dataset."""
    cols_for_row_id: list[str]
    fingerprint: str

class ProcessedDatasetMetadata(SharedInterimProcessedMetadata):
    """Schema for the metadata of a processed dataset."""
    processed_run_id: str
    row_id_info: RowIdInfo | None = None
