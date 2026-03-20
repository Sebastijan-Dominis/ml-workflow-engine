"""A module defining the data models for snapshot bindings configuration."""

from pydantic import BaseModel, Field, RootModel


class DatasetSnapshotBinding(BaseModel):
    """Defines the snapshot binding for a specific dataset version."""
    snapshot: str = Field(..., description="The identifier of the snapshot to use for this dataset version.")


class FeatureSetSnapshotBinding(BaseModel):
    """Defines the snapshot binding for a specific feature set version."""
    snapshot: str = Field(..., description="The identifier of the snapshot to use for this feature set version.")


class SnapshotBinding(BaseModel):
    """Defines the overall snapshot binding configuration, including dataset and feature set bindings, with versions."""
    datasets: dict[str, dict[str, DatasetSnapshotBinding]] = Field(
        default_factory=dict,
        description=(
            "Mapping of dataset names to versions, which then map to their respective snapshot bindings.\n"
            "Example: datasets[dataset_name][dataset_version].snapshot"
        )
    )
    feature_sets: dict[str, dict[str, FeatureSetSnapshotBinding]] = Field(
        default_factory=dict,
        description=(
            "Mapping of feature set names to versions, which then map to their respective snapshot bindings.\n"
            "Example: feature_sets[feature_set_name][feature_set_version].snapshot"
        )
    )


class SnapshotBindingsRegistry(RootModel[dict[str, SnapshotBinding]]):
    """Defines the registry of snapshot bindings, mapping binding keys to their configurations."""
    def __getitem__(self, key: str) -> SnapshotBinding:
        return self.root[key]

    def get(self, key: str, default=None):
        return self.root.get(key, default)
