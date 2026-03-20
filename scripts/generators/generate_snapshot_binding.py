"""Utility CLI to generate snapshot bindings for datasets and feature sets.

This script scans the processed datasets and feature sets to find the latest snapshots,
and then creates a timestamped binding entry in the snapshot bindings registry YAML.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml

from ml.exceptions import DataError
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.logging_config import setup_logging
from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path

logger = logging.getLogger(__name__)

# Base directories
DATA_PROCESSED_DIR = Path("data/processed")
FEATURE_STORE_DIR = Path("feature_store")
BINDINGS_PATH = Path("configs/snapshot_bindings_registry/bindings.yaml")


def scan_latest_snapshots(base_dir: Path) -> dict[str, dict[str, str]]:
    """Scan a base directory for latest snapshots.

    Returns:
        dict[name][version] = snapshot
    """
    result = {}
    if not base_dir.exists():
        logger.warning(f"Base directory {base_dir} does not exist. Skipping.")
        return result

    for name_dir in base_dir.iterdir():
        if not name_dir.is_dir():
            continue
        result[name_dir.name] = {}
        for version_dir in name_dir.iterdir():
            if not version_dir.is_dir():
                continue
            try:
                latest_snapshot = get_latest_snapshot_path(version_dir)
                result[name_dir.name][version_dir.name] = latest_snapshot.name
            except DataError:
                logger.warning(f"No valid snapshots found for {version_dir}. Skipping.")
    return result


def load_bindings(path: Path) -> dict:
    """Load existing bindings YAML or create an empty dict."""
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def save_bindings_atomic(path: Path, data: dict):
    """Save bindings dictionary to YAML atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=True)
    temp_path.replace(path)  # atomic replace

def main() -> int:
    """Generate snapshot bindings and update bindings.yaml."""
    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    log_file = Path(f"scripts_logs/generators/generate_snapshot_binding/{run_id}/snapshot_binding_generation.log")
    setup_logging(path=log_file, level=logging.INFO)

    logger.info(f"Starting snapshot binding generation with run_id {run_id}")

    try:
        datasets = scan_latest_snapshots(DATA_PROCESSED_DIR)
        feature_sets = scan_latest_snapshots(FEATURE_STORE_DIR)
        logger.info(f"Found datasets: {datasets}")
        logger.info(f"Found feature sets: {feature_sets}")

        bindings = load_bindings(BINDINGS_PATH)

        bindings[run_id] = {
            "datasets": {name: {version: {"snapshot": snap} for version, snap in versions.items()}
                         for name, versions in datasets.items()},
            "feature_sets": {name: {version: {"snapshot": snap} for version, snap in versions.items()}
                             for name, versions in feature_sets.items()}
        }

        save_bindings_atomic(BINDINGS_PATH, bindings)
        logger.info(f"Snapshot bindings successfully written to {BINDINGS_PATH} under key {run_id}")
        print(f"Snapshot bindings successfully written to {BINDINGS_PATH} under key {run_id}")
        return 0

    except Exception as e:
        logger.exception(f"Failed to generate snapshot bindings: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())