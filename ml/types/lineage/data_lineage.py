"""Data models describing dataset lineage and validation metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataLineageEntry:
    """Normalized lineage record for a dataset included in a merged artifact."""

    ref: str
    name: str
    version: str
    format: str
    path_suffix: str
    merge_key: str
    snapshot_id: str
    path: str
    loader_validation_hash: str
    data_hash: str
    row_count: int
    column_count: int
