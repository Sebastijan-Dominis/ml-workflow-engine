import logging
from pathlib import Path
from typing import Optional

from ml.exceptions import DataError
from ml.utils.features.loading.latest_snapshot import get_latest_snapshot
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def resolve_feature_snapshots(
    feature_store_path: Path,
    feature_sets: list,
    snapshot_binding: Optional[list[dict]] = None
) -> list[dict]:
    """
    Resolve the snapshot paths to load for each feature set.

    Args:
        feature_store_path: Base path to feature store
        feature_sets: List of feature set specs from model config
        snapshot_binding: Optional. If provided, a list of dicts containing
                          pre-selected snapshot_ids for each feature set (from train metadata)

    Returns:
        List of dicts with keys:
            - fs_spec: the feature set spec
            - snapshot_path: Path object to the snapshot
            - snapshot_id: the snapshot folder name
            - metadata: loaded metadata.json
    """
    resolved = []

    for i, fs in enumerate(feature_sets):
        version_path = feature_store_path / fs.ref.replace(".", "/") / fs.name / fs.version

        if snapshot_binding:
            # Use provided snapshot (eval/explain)
            snapshot_id = snapshot_binding[i]["snapshot_id"]
            snapshot_path = version_path / snapshot_id
        else:
            # Default: latest snapshot (train/search)
            snapshot_path = get_latest_snapshot(version_path)
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