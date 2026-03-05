"""Schemas for interim dataset metadata."""
from ml.metadata.schemas.data.shared import SharedInterimProcessedMetadata


class InterimDatasetMetadata(SharedInterimProcessedMetadata):
    """Schema for the metadata of an interim dataset."""
    interim_run_id: str
