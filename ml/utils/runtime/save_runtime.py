import json
import logging
from pathlib import Path

from ml.config.validation_schemas.hardware_cfg import HardwareConfig
from ml.exceptions import PersistenceError
from ml.utils.runtime.runtime_snapshot import build_runtime_snapshot

logger = logging.getLogger(__name__)

def save_runtime_snapshot(
    *, 
    target_dir: Path, 
    timestamp: str, 
    hardware_info: HardwareConfig, 
    start_time: float,
    overwrite_existing: bool = False
) -> None:
    snapshot = build_runtime_snapshot(timestamp, hardware_info, start_time=start_time)
    snapshot_path = target_dir / "runtime.json"

    if snapshot_path.exists():
        if not overwrite_existing:
            msg = f"Runtime snapshot file already exists at {snapshot_path}, and overwriting is disabled. Set overwrite_existing=True to overwrite it."
            logger.error(msg)
            raise PersistenceError(msg)
        logger.critical(f"Runtime snapshot file already exists at {snapshot_path}, and overwriting is enabled. It will be overwritten.")

    try:
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=4, sort_keys=True, default=str)
        logger.info("Runtime snapshot successfully saved to %s", snapshot_path)
    except Exception as e:
        msg = f"Failed to save runtime snapshot to {snapshot_path}"
        logger.exception(msg)
        raise PersistenceError(msg) from e
