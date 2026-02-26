import logging
from pathlib import Path

from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path

logger = logging.getLogger(__name__)

def get_snapshot_path(snapshot_id: str, snapshot_dir: Path) -> Path:
    if snapshot_id == "latest":
        latest_snapshot = get_latest_snapshot_path(snapshot_dir)
        logger.info(f"Auto-resolved latest snapshot ID: {latest_snapshot}")
        return latest_snapshot
    return snapshot_dir / snapshot_id