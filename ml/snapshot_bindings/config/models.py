"""A module defining the data models for snapshot bindings configuration."""

from pydantic import BaseModel, Field, RootModel


class DatasetSnapshotBinding(BaseModel):
    """Defines the snapshot binding for a specific dataset."""
    snapshot: str = Field(..., description="The identifier of the snapshot to use for this dataset.")

# Intentionally redundant with DatasetSnapshotBinding for clarity and potential future divergence in structure or metadata.
class FeatureSetSnapshotBinding(BaseModel):
    """Defines the snapshot binding for a specific feature set."""
    snapshot: str = Field(..., description="The identifier of the snapshot to use for this feature set.")

# Datasets and feature sets intentionally not required at the SnapshotBinding level to allow for flexible configurations that may include only one type of binding. Validation will ensure that if a snapshot binding is used, the necessary dataset or feature set bindings are present.
class SnapshotBinding(BaseModel):
    """Defines the overall snapshot binding configuration, including dataset and feature bindings."""
    datasets: dict[str, DatasetSnapshotBinding] = Field(
        default_factory=dict,
        description="Mapping of dataset names to their respective snapshot bindings."
    )
    feature_sets: dict[str, FeatureSetSnapshotBinding] = Field(
        default_factory=dict,
        description="Mapping of feature set names to their respective snapshot bindings."
    )

class SnapshotBindingsRegistry(RootModel[dict[str, SnapshotBinding]]):
    """Defines the registry of snapshot bindings, mapping binding keys to their configurations."""
    def __getitem__(self, key: str) -> SnapshotBinding:
        return self.root[key]

    def get(self, key: str, default=None):
        return self.root.get(key, default)
