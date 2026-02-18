import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from ml.cli.error_handling import resolve_exit_code
from ml.data.interim.data_preparation.prepare_data import (clean_data,
                                                           enforce_schema,
                                                           normalize_columns)
from ml.data.interim.persistence.prepare_metadata import prepare_metadata
from ml.data.interim.validation.validate import \
    validate_min_rows_after_cleaning
from ml.data.utils.config.schemas.interim import InterimConfig
from ml.data.utils.config.validate_config import validate_config
from ml.data.utils.memory.compute_memory_change import compute_memory_change
from ml.data.utils.memory.get_memory_usage import get_memory_usage
from ml.data.utils.persistence.save_data import save_data
from ml.exceptions import DataError, UserError
from ml.logging_config import setup_logging
from ml.utils.data.validate_dataset import validate_dataset
from ml.utils.loaders import load_json, load_yaml, read_data
from ml.utils.persistence.save_metadata import save_metadata

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make interim dataset by cleaning and transforming raw data according to the provided configuration.")

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

def main() -> int:
    args: argparse.Namespace
    config_raw: dict
    config: InterimConfig
    df: pd.DataFrame
    dataset_path: Path
    metadata: dict

    args = parse_args()

    start_time = time.perf_counter()

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    data_dir = Path("data/interim") / args.dataset / args.version
    setup_logging(data_dir / "make_interim.log", level=log_level)

    try:
        config_raw = load_yaml(Path(f"configs/data/interim/{args.dataset}/{args.version}.yaml"))
        config = validate_config(config_raw, type="interim")

        data_path = data_dir / config.dataset.output.path_suffix
        if data_path.exists():
            msg = f"Data file {data_path} already exists. Aborting to prevent overwriting."
            logger.error(msg)
            raise UserError(msg)

        raw_metadata = load_json(Path(f"data/raw/{args.dataset}/{args.version}/metadata.json"))

        validate_dataset(data_path=Path(config.input.path), metadata=raw_metadata)

        df = read_data(config.input.format, Path(config.input.path))

        df = normalize_columns(df, config.cleaning)
        df = enforce_schema(df, schema=config.data_schema, drop_missing_ints=config.drop_missing_ints)
        df = clean_data(df, config.invariants)
        validate_min_rows_after_cleaning(df, config.min_rows)

        if config.drop_duplicates:
            df = df.drop_duplicates()

        dataset_path = save_data(df, config=config, data_dir=data_dir)

        logger.info(f"Interim dataset created successfully at {dataset_path} with {len(df)} rows and {len(df.columns)} columns.")

        memory_usage = get_memory_usage(df)

        memory_info = compute_memory_change(target_metadata=raw_metadata, new_memory_usage=memory_usage, stage="interim")

        metadata = prepare_metadata(
            df, 
            config=config, 
            start_time=start_time,
            dataset_path=dataset_path,
            owner=args.owner,
            memory_info=memory_info
        )

        save_metadata(metadata, target_dir=data_dir)

        return 0
    
    except Exception as e:
        exit_code = resolve_exit_code(e)
        return exit_code

if __name__ == "__main__":
    sys.exit(main())