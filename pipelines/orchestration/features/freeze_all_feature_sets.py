"""Bulk feature freezing orchestrator.

This script iterates through feature registry entries and executes the feature
freeze pipeline for each feature-set version, with optional skipping when
existing snapshot folders are present.
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.formatting.str_to_bool import str_to_bool
from ml.logging_config import setup_logging
from ml.utils.loaders import load_yaml

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for bulk feature freezing.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Freeze features.")

    parser.add_argument(
        "--logging-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    parser.add_argument(
        "--owner",
        type=str,
        default="Sebastijan",
        help="Owner of the feature sets (default: Sebastijan)"
    )

    parser.add_argument(
        "--skip-if-existing",
        type=str_to_bool,
        default=True,
        help="Skip freezing if at least one freeze folder already exists for the feature set (default: True)"
    )

    return parser.parse_args()

def log_completion(start_time: float, message: str):
    """Log script completion timing details.

    Args:
        start_time: Start time from ``time.perf_counter()``.
        message: Completion message to emit.
    """
    end_time = time.perf_counter()
    duration = end_time - start_time
    end = iso_no_colon(datetime.now())
    logger.info(f"{message} at {end} after {duration:.2f} seconds")

def main() -> int:
    """Freeze all feature sets registered in the feature registry.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Existing freeze directories can be skipped for idempotent runs when
        ``--skip-if-existing`` is enabled.

    Side Effects:
        Executes feature-freezing subprocess calls per registry entry and writes
        batch-level logs.

    Examples:
        python -m pipelines.orchestration.features.freeze_all_feature_sets --skip-if-existing true
    """
    args = parse_args()

    start_time = time.perf_counter()

    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    log_file = Path(f"orchestration_logs/features/freeze_all_feature_sets/{run_id}/freeze_all.log")
    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    setup_logging(path=log_file, level=log_level)

    logger.info(f"Script started at {timestamp} with run ID {run_id}, logging level {args.logging_level.upper()} and owner {args.owner}")

    successes_count = 0

    try:
        feature_registry = load_yaml(Path("configs/feature_registry/features.yaml"))
    except Exception:
        logger.exception("Failed to load feature registry.")
        return 1

    features_root = Path("feature_store")

    for feature_set_name in feature_registry:
        for feature_set_version in feature_registry[feature_set_name]:

            freeze_dir = features_root / feature_set_name / feature_set_version

            existing_freezes = (
                [d for d in freeze_dir.iterdir() if d.is_dir()]
                if freeze_dir.exists()
                else []
            )

            if existing_freezes and args.skip_if_existing:
                logger.info(
                    f"Skipping '{feature_set_name}' v{feature_set_version} "
                    f"because freeze folders already exist: {[d.name for d in existing_freezes]}"
                )
                continue

            cmd = [
                sys.executable,
                "-m", "pipelines.features.freeze",
                "--feature-set", feature_set_name,
                "--version", feature_set_version,
                "--logging-level", args.logging_level.upper(),
                "--owner", args.owner
            ]

            try:
                logger.info(f"Freezing feature set '{feature_set_name}' version '{feature_set_version}'...")
                subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
                logger.info(f"Feature set '{feature_set_name}' succeeded.")
                successes_count += 1
            except subprocess.CalledProcessError as e:
                first_line = e.stderr.splitlines()[0] if e.stderr else ""
                logger.error(f"Failed to freeze '{feature_set_name}' version '{feature_set_version}': {first_line}")
                log_completion(start_time, f"Script terminated after successfully freezing {successes_count} feature sets")
                return e.returncode

    log_completion(start_time, f"Script completed successfully after freezing {successes_count} feature sets")
    return 0

if __name__ == "__main__":
    sys.exit(main())
