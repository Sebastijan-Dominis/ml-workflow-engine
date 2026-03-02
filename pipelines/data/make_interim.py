"""CLI for building interim datasets from raw snapshots.

This pipeline loads raw data, applies schema and cleaning rules defined in
configuration, writes the interim dataset, and persists run metadata.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ml.cli.error_handling import resolve_exit_code
from ml.data.config.schemas.interim import InterimConfig
from ml.data.config.validate_config import validate_config
from ml.data.interim.data_preparation.prepare_data import (clean_data,
                                                           enforce_schema,
                                                           normalize_columns)
from ml.data.interim.persistence.prepare_metadata import prepare_metadata
from ml.data.utils.memory.compute_memory_change import compute_memory_change
from ml.data.utils.memory.get_memory_usage import get_memory_usage
from ml.data.utils.persistence.save_data import save_data
from ml.logging_config import setup_logging
from ml.utils.data.get_data_suffix_and_format import get_data_suffix_and_format
from ml.utils.data.validate_data import validate_data
from ml.utils.data.validate_min_rows import validate_min_rows
from ml.utils.formatting.iso_no_col import iso_no_colon
from ml.utils.loaders import load_json, load_yaml, read_data
from ml.utils.persistence.save_metadata import save_metadata
from ml.utils.snapshots.snapshot_path import get_snapshot_path

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for interim data creation.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Make interim data by cleaning and transforming raw data according to the provided configuration.")

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
        "--raw-snapshot-id",
        type=str,
        default="latest",
        help="Snapshot ID for the raw data version to use (optional; defaults to latest if not provided)"
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
        help="Owner of the data (default: Sebastijan)"
    )

    return parser.parse_args()

def main() -> int:
    """Execute the interim data preparation workflow.

    The workflow loads and validates interim configuration, reads source raw
    data, applies normalization/schema/cleaning steps, writes the transformed
    dataset, and persists metadata for reproducibility.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Exceptions are converted to process exit codes; the function is designed
        as a CLI boundary and does not propagate failures upward.

    Side Effects:
        Creates a new interim snapshot directory, writes transformed dataset
        artifacts, metadata, and run logs.

    Examples:
        python pipelines/data/make_interim.py --data hotel_bookings --version v1 --raw-snapshot-id latest
    """
    args: argparse.Namespace
    config_raw: dict
    config: InterimConfig
    df: pd.DataFrame
    data_path: Path
    metadata: dict

    args = parse_args()

    start_time = time.perf_counter()

    timestamp = iso_no_colon(datetime.now())
    interim_id = f"{timestamp}_{uuid4().hex[:8]}"
    
    data_dir = Path("data/interim") / args.data / args.version / interim_id
    data_dir.mkdir(parents=True, exist_ok=False)

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    setup_logging(data_dir / "make_interim.log", level=log_level)

    try:
        config_raw = load_yaml(Path(f"configs/data/interim/{args.data}/{args.version}.yaml"))
        config = validate_config(config_raw, type="interim")

        raw_data_dir = Path("data/raw") / args.data / config.raw_data_version

        raw_data_snapshot_path = get_snapshot_path(args.raw_snapshot_id, raw_data_dir)

        raw_metadata = load_json(raw_data_snapshot_path / "metadata.json")

        raw_data_suffix, raw_data_format = get_data_suffix_and_format(raw_metadata, location="data")
        
        raw_data_path = raw_data_snapshot_path / raw_data_suffix
        validate_data(data_path=raw_data_path, metadata=raw_metadata)
        logger.debug(f"Validated raw data at {raw_data_path} with metadata: {raw_metadata}")

        df = read_data(raw_data_format, raw_data_path)
        logger.debug(f"Read raw data into DataFrame with {len(df)} rows and {len(df.columns)} columns.")

        df = normalize_columns(df, config.cleaning)
        logger.debug(f"Normalized columns. DataFrame now has columns: {df.columns.tolist()}.")

        df = enforce_schema(df, schema=config.data_schema, drop_missing_ints=config.drop_missing_ints)
        logger.debug(f"Enforced schema. DataFrame now has columns: {df.columns.tolist()} with dtypes: {df.dtypes.astype(str).to_dict()}.")
        
        df = clean_data(df, config.invariants)
        logger.debug(f"Cleaned data using invariants. DataFrame now has {len(df)} rows and {len(df.columns)} columns.")
        
        validate_min_rows(df, config.min_rows)

        if config.drop_duplicates:
            df = df.drop_duplicates()
            logger.info(f"Dropped duplicates. DataFrame now has {len(df)} rows.")

        data_path = save_data(df, config=config, data_dir=data_dir)
        logger.info(f"Interim data saved successfully at {data_path} with {len(df)} rows and {len(df.columns)} columns.")

        memory_usage = get_memory_usage(df)

        memory_info = compute_memory_change(target_metadata=raw_metadata, new_memory_usage=memory_usage, stage="interim")

        metadata = prepare_metadata(
            df, 
            config=config, 
            start_time=start_time,
            data_path=data_path,
            source_data_path=raw_data_path,
            source_data_format=raw_data_format,
            owner=args.owner,
            memory_info=memory_info,
            interim_run_id=interim_id
        )

        save_metadata(metadata, target_dir=data_dir)

        return 0
    
    except Exception as e:
        exit_code = resolve_exit_code(e)
        return exit_code

if __name__ == "__main__":
    sys.exit(main())