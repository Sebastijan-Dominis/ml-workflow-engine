import logging
logger = logging.getLogger(__name__)
import json
from pathlib import Path
from datetime import datetime

from ml.utils.runtime.runtime_snapshot import build_runtime_snapshot
from ml.exceptions import PersistenceError

def save_runtime_snapshot(run_dir: Path, timestamp: str) -> None:
    snapshot = build_runtime_snapshot(timestamp)
    snapshot_path = run_dir / "runtime.json"
    try:
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=4, sort_keys=True, default=str)
        logger.info("Saved runtime snapshot to %s", snapshot_path)
    except Exception as e:
        msg = f"Failed to save runtime snapshot to {snapshot_path}"
        logger.exception(msg)
        raise PersistenceError(msg) from e
