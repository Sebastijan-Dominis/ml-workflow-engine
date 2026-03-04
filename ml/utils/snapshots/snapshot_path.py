"""Utilities for resolving explicit or latest snapshot directory paths."""

import logging
from pathlib import Path

from ml.exceptions import RuntimeMLError
from ml.types import LatestSnapshot
from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path

logger = logging.getLogger(__name__)

def get_snapshot_path(snapshot_id: str, snapshot_dir: Path) -> Path:
    """Resolve snapshot path from ID, supporting automatic latest selection.

    Args:
        snapshot_id: Snapshot identifier or ``latest``.
        snapshot_dir: Directory containing snapshot subdirectories.

    Returns:
        Resolved snapshot path.
    """

    if snapshot_id == LatestSnapshot.LATEST.value:
        try:
            latest_snapshot = get_latest_snapshot_path(snapshot_dir)
        except Exception as e:
            msg = f"Failed to resolve latest snapshot in directory {snapshot_dir}. Please ensure that the directory exists and contains valid snapshot folders with names in the format 'timestamp_uuid', e.g., '20240101T120000_abcdef12'."
            logger.exception(msg)
            raise RuntimeMLError(msg) from e

        logger.info(f"Auto-resolved latest snapshot ID: {latest_snapshot}")
        return latest_snapshot
    return snapshot_dir / snapshot_id
