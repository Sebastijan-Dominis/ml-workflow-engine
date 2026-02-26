import argparse
import logging
import sys
from pathlib import Path

from ml.cli.error_handling import resolve_exit_code
from ml.data.raw.persistence.prepare_metadata import prepare_metadata
from ml.exceptions import UserError
from ml.logging_config import add_file_handler, bootstrap_logging
from ml.utils.loaders import read_data
from ml.utils.persistence.save_metadata import save_metadata
from ml.utils.snapshots.snapshot_path import get_snapshot_path

logger = logging.getLogger(__name__)

def parse_args():
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
        "--format", 
        type=str, 
        required=True, 
        help="Format of the data (e.g., csv, parquet)."
    )

    parser.add_argument(
        "--snapshot_id",
        type=str,
        default="latest",
        help="Snapshot ID for tracking the data version (optional; defaults to latest if not provided)"
    )

    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    parser.add_argument(
        "--owner",
        type=str,
        default="Sebastijan",
        help="Owner of the experiment (default: Sebastijan)"
    )

    return parser.parse_args()

def main():
    args: argparse.Namespace

    args = parse_args()

    data_parent_dir = Path("data/raw") / args.data / args.version

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    bootstrap_logging(level=log_level)

    try:
        data_dir = get_snapshot_path(args.snapshot_id, data_parent_dir)
    except Exception as e:
        logging.exception("Failed to get snapshot path")
        return resolve_exit_code(e)

    log_path = data_dir / "handle_raw.log"
    add_file_handler(log_path, level=log_level)
    try:

        data_suffix = f"data.{args.format}"
        data_path = data_dir / data_suffix
        df = read_data(args.format, data_path)

        metadata = prepare_metadata(
            df, 
            args=args, 
            data_path=data_path, 
            raw_run_id=data_dir.name,
            data_suffix=data_suffix
        )

        save_metadata(metadata, target_dir=data_dir)

        return 0
    
    except Exception as e:
        exit_code = resolve_exit_code(e)
        return exit_code
    
if __name__ == "__main__":
    sys.exit(main())