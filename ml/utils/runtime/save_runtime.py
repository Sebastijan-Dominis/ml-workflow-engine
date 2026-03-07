"""Persistence utilities for writing runtime snapshot artifacts."""

import json
import logging
import os
import tempfile
from pathlib import Path

from ml.config.schemas.hardware_cfg import HardwareConfig
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
    """Build and persist runtime snapshot JSON with overwrite protections."""

    snapshot = build_runtime_snapshot(timestamp, hardware_info, start_time=start_time)
    snapshot_path = target_dir / "runtime.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    if snapshot_path.exists():
        if not overwrite_existing:
            msg = f"Runtime snapshot file already exists at {snapshot_path}, and overwriting is disabled. Set overwrite_existing=True to overwrite it."
            logger.error(msg)
            raise PersistenceError(msg)
        logger.critical(f"Runtime snapshot file already exists at {snapshot_path}, and overwriting is enabled. It will be overwritten.")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=snapshot_path.parent,
            prefix="runtime.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            temp_path = tmp_file.name
            json.dump(snapshot, tmp_file, indent=4, sort_keys=True, default=str)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(temp_path, snapshot_path)
        logger.info("Runtime snapshot successfully saved to %s", snapshot_path)
    except Exception as e:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except OSError:
                logger.warning("Failed to clean up temporary runtime snapshot file: %s", temp_path)

        msg = f"Failed to save runtime snapshot to {snapshot_path}"
        logger.exception(msg)
        raise PersistenceError(msg) from e
