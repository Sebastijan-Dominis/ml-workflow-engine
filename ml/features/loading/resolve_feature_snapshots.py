"""Helpers for resolving feature snapshot directories for data loading."""

import logging
from pathlib import Path

from ml.exceptions import ConfigError
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.snapshot_bindings.extraction.get_snapshot_binding import get_and_validate_snapshot_binding
from ml.utils.loaders import load_json
from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path

logger = logging.getLogger(__name__)

def resolve_feature_snapshots(
    feature_store_path: Path,
    feature_sets: list,
    snapshot_binding: list[FeatureLineage] | None = None,
    snapshot_binding_key: str | None = None
) -> list[dict]:
    """Resolve snapshot metadata for each feature set using binding or latest snapshot.

    Args:
        feature_store_path: Root path of the feature store.
        feature_sets: Configured feature-set specifications.
        snapshot_binding: Optional explicit snapshot mapping, typically from training metadata.
        snapshot_binding_key: Optional key for a snapshot binding to define which snapshot to load for each dataset.
    Returns:
        list[dict]: Resolved snapshot descriptors containing feature spec, path, id, and metadata.
    """

    snapshot_binding_config = None
    if snapshot_binding_key:
        snapshot_binding_config = get_and_validate_snapshot_binding(
            snapshot_binding_key,
            expect_feature_set_bindings=True
        )

    resolved = []

    for i, fs in enumerate(feature_sets):
        version_path = feature_store_path / fs.name / fs.version
        if snapshot_binding_config:
            feature_sets_binding = snapshot_binding_config.feature_sets
            feature_set_binding = feature_sets_binding.get(fs.name) if feature_sets_binding else None
            if not feature_set_binding:
                msg = f"Snapshot binding for feature set {fs.name} not found in snapshot binding configuration."
                logger.error(msg)
                raise ConfigError(msg)
            snapshot_id = feature_set_binding.snapshot
            snapshot_path = version_path / snapshot_id
        elif snapshot_binding:
            # Use provided snapshot (eval/explain)
            snapshot_id = snapshot_binding[i].snapshot_id
            snapshot_path = version_path / snapshot_id
        else:
            # Default: latest snapshot (train/search)
            snapshot_path = get_latest_snapshot_path(version_path)
            snapshot_id = snapshot_path.name

        logger.debug(f"Resolving feature set {fs.name} {fs.version} to snapshot {snapshot_path}")

        metadata_path = snapshot_path / "metadata.json"
        metadata = load_json(metadata_path)

        resolved.append({
            "fs_spec": fs,
            "snapshot_path": snapshot_path,
            "snapshot_id": snapshot_id,
            "metadata": metadata
        })

    return resolved
