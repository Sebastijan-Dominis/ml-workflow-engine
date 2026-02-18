import argparse
import logging
import sys
from pathlib import Path

from ml.cli.error_handling import resolve_exit_code
from ml.data.raw.persistence.prepare_metadata import prepare_metadata
from ml.exceptions import UserError
from ml.logging_config import setup_logging
from ml.utils.loaders import read_data
from ml.utils.persistence.save_metadata import save_metadata

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Handle raw data for the hotel management ML project.")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g., 'hotel_bookings'"
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Dataset version, e.g., 'v1'"
    )

    parser.add_argument(
        "--path-suffix",
        type=str,
        required=True,
        help="Name of the path suffix, e.g., 'hotel_bookings_kaggle.csv'"
    )

    parser.add_argument(
        "--format", 
        type=str, 
        required=True, 
        help="Format of the dataset (e.g., csv, parquet)."
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
    data_dir = Path("data/raw") / args.dataset / args.version
    data_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(data_dir / "handle_raw.log", level=logging.INFO)

    try:
        if (data_dir / "metadata.json").exists():
            msg = f"Metadata already exists for dataset {args.dataset} version {args.version}. Aborting to prevent overwrite."
            logger.error(msg)
            raise UserError(msg)
        dataset_path = data_dir / args.path_suffix
        df = read_data(args.format, dataset_path)

        metadata = prepare_metadata(df, args=args, dataset_path=dataset_path)

        save_metadata(metadata, target_dir=data_dir)

        return 0
    
    except Exception as e:
        exit_code = resolve_exit_code(e)
        return exit_code
    
if __name__ == "__main__":
    sys.exit(main())