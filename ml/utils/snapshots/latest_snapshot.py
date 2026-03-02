"""Utilities for resolving the latest valid snapshot directory."""

import logging
from datetime import datetime
from pathlib import Path

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def get_latest_snapshot_path(snapshot_dir: Path) -> Path:
    """Return newest valid snapshot path using timestamp and UUID tie-breaking.

    Args:
        snapshot_dir: Directory containing snapshot subdirectories.

    Returns:
        Path: Latest valid snapshot path.
    """

    snapshots = []
    
    for s in snapshot_dir.iterdir():
        if not s.is_dir():
            continue
        parts = s.name.split("_")
        if len(parts) != 2:
            logger.warning(f"Ignoring folder with unexpected format: {s.name}")
            continue
        timestamp_str, uuid_str = parts
        if 'T' not in timestamp_str:
            logger.warning(f"Ignoring folder with invalid timestamp: {s.name}")
            continue
        snapshots.append(s)
    
    if not snapshots:
        msg = f"No valid snapshots found in {snapshot_dir}"
        logger.error(msg)
        raise DataError(msg)

    def parse_snapshot(p: Path) -> tuple[datetime, str]:
        """Parse snapshot folder name into comparable datetime and UUID parts.

        Args:
            p: Snapshot directory path.

        Returns:
            tuple[datetime, str]: Parsed timestamp and UUID components.
        """

        timestamp_str, uuid_str = p.name.split("_")
        date_part, time_part = timestamp_str.split('T')
        time_part = time_part.replace('-', ':')
        dt = datetime.fromisoformat(f"{date_part}T{time_part}")
        return (dt, uuid_str)

    latest_snapshot = max(snapshots, key=parse_snapshot)

    latest_dt, _ = parse_snapshot(latest_snapshot)
    tied_snapshots = [
        p for p in snapshots if parse_snapshot(p)[0] == latest_dt
    ]
    if len(tied_snapshots) > 1:
        tied_names = [p.name for p in tied_snapshots]
        logger.warning(
            f"Multiple snapshots have the same timestamp {latest_dt.isoformat()}: {tied_names}. "
            "Tie-breaking was done using UUIDs, which may not reflect true creation order."
        )

    return latest_snapshot