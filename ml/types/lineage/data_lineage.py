"""Data models describing dataset lineage and validation metadata."""

from dataclasses import dataclass

from ml.feature_freezing.freeze_strategies.tabular.config.models import MergeHow, MergeValidate


@dataclass(frozen=True)
class DataLineageEntry:
    """Normalized lineage record for a dataset included in a merged artifact."""

    ref: str
    name: str
    version: str
    format: str
    path_suffix: str
    merge_key: tuple[str, ...] | str
    merge_how: MergeHow
    merge_validate: MergeValidate
    snapshot_id: str
    path: str
    loader_validation_hash: str
    data_hash: str
    row_count: int
    column_count: int

    def __post_init__(self):
        if isinstance(self.merge_key, list):
            object.__setattr__(self, "merge_key", tuple(self.merge_key))
