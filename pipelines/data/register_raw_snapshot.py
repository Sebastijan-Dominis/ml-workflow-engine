"""CLI for validating and recording metadata for raw data snapshots.

This module resolves a raw snapshot path, loads the snapshot's single data file,
prepares metadata, and persists that metadata alongside the snapshot.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from ml.cli.error_handling import resolve_exit_code
from ml.data.raw.persistence.prepare_metadata import prepare_metadata
from ml.exceptions import UserError
from ml.io.persistence.save_metadata import save_metadata
from ml.logging_config import add_file_handler, bootstrap_logging
from ml.metadata.schemas.data.raw import RawSnapshotMetadata
from ml.types import LatestSnapshot
from ml.utils.loaders import read_data
from ml.utils.snapshots.snapshot_path import get_snapshot_path

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the raw data handling pipeline.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Handle raw data for the hotel management ML project.")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Data name, e.g., 'hotel_bookings'"
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Data version, e.g., 'v1'"
    )

    parser.add_argument(
        "--snapshot_id",
        type=str,
        default=LatestSnapshot.LATEST.value,
        help="Snapshot ID for tracking the data version (optional; defaults to latest if not provided)"
    )

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
        help="Owner of the metadata (default: Sebastijan)"
    )

    return parser.parse_args()

def main() -> int:
    """Run the raw data handling workflow.

    The workflow resolves the requested raw snapshot, validates that exactly one
    data file is present, reads the file, builds metadata, and persists
    `metadata.json` into the snapshot directory.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Exceptions are translated to standardized CLI exit codes via
        ``resolve_exit_code`` rather than being re-raised.

    Side Effects:
        Creates/updates ``register_raw_snapshot.log`` and writes ``metadata.json`` to the
        selected raw snapshot directory.

    Examples:
        python pipelines/data/register_raw_snapshot.py --data hotel_bookings --version v1 --snapshot_id latest
    """
    args: argparse.Namespace
    metadata: RawSnapshotMetadata
    df: pd.DataFrame

    args = parse_args()

    data_parent_dir = Path("data/raw") / args.data / args.version

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    bootstrap_logging(level=log_level)

    try:
        data_dir = get_snapshot_path(args.snapshot_id, data_parent_dir)
    except Exception as e:
        logging.exception("Failed to get snapshot path")
        return resolve_exit_code(e)

    log_path = data_dir / "register_raw_snapshot.log"
    add_file_handler(log_path, level=log_level)
    try:
        data_files = list(data_dir.glob("data.*"))

        if len(data_files) != 1:
            raise UserError(
                f"Expected exactly one data file in {data_dir}, found {len(data_files)}"
            )

        data_path = data_files[0]
        data_suffix = data_path.name
        data_format = data_path.suffix.lstrip(".")

        df = read_data(data_format, data_path)

        metadata = prepare_metadata(
            df,
            args=args,
            data_path=data_path,
            raw_run_id=data_dir.name,
            data_format=data_format,
            data_suffix=data_suffix
        )

        save_metadata(metadata.model_dump(exclude_none=True), target_dir=data_dir)

        return 0

    except Exception as e:
        exit_code = resolve_exit_code(e)
        return exit_code

if __name__ == "__main__":
    sys.exit(main())
