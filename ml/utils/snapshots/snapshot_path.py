import logging
from pathlib import Path

from ml.exceptions import RuntimeMLException
from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path

logger = logging.getLogger(__name__)

def get_snapshot_path(snapshot_id: str, snapshot_dir: Path) -> Path:
    if snapshot_id == "latest":
        try:
            latest_snapshot = get_latest_snapshot_path(snapshot_dir)
        except Exception:
            msg = f"Failed to resolve latest snapshot in directory {snapshot_dir}. Please ensure that the directory exists and contains valid snapshot folders with names in the format 'timestamp_uuid', e.g., '20240101T120000_abcdef12'."
            logger.error(msg)
            raise RuntimeMLException(msg)
        
        logger.info(f"Auto-resolved latest snapshot ID: {latest_snapshot}")
        return latest_snapshot
    return snapshot_dir / snapshot_id