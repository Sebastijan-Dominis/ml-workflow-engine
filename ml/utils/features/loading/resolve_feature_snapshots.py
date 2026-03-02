"""Helpers for resolving feature snapshot directories for data loading."""

# TODO: Adjust the rest of the code to actually enable snapshot binding
import logging
from pathlib import Path
from typing import Optional

from ml.exceptions import DataError
from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def resolve_feature_snapshots(
    feature_store_path: Path,
    feature_sets: list,
    snapshot_binding: Optional[list[dict]] = None
) -> list[dict]:
    """Resolve snapshot metadata for each feature set using binding or latest snapshot.

    Args:
        feature_store_path: Root path of the feature store.
        feature_sets: Configured feature-set specifications.
        snapshot_binding: Optional explicit snapshot mapping, typically from training metadata.

    Returns:
        list[dict]: Resolved snapshot descriptors containing feature spec, path, id, and metadata.
    """

    resolved = []

    for i, fs in enumerate(feature_sets):
        version_path = feature_store_path / fs.name / fs.version

        if snapshot_binding:
            # Use provided snapshot (eval/explain)
            snapshot_id = snapshot_binding[i]["snapshot_id"]
            snapshot_path = version_path / snapshot_id
        else:
            # Default: latest snapshot (train/search)
            snapshot_path = get_latest_snapshot_path(version_path)
            snapshot_id = snapshot_path.name

        logger.debug(f"Resolving feature set {fs.name} {fs.version} to snapshot {snapshot_path}")

        metadata_path = snapshot_path / "metadata.json"
        if not metadata_path.exists():
            msg = f"Missing metadata.json in snapshot path {snapshot_path}"
            logger.error(msg)
            raise DataError(msg)

        metadata = load_json(metadata_path)

        resolved.append({
            "fs_spec": fs,
            "snapshot_path": snapshot_path,
            "snapshot_id": snapshot_id,
            "metadata": metadata
        })

    return resolved