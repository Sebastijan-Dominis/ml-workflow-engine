import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ml.cli.error_handling import resolve_exit_code
from ml.data.processed.persistence.prepare_metadata import prepare_metadata
from ml.data.processed.processing.process_data import (add_row_id,
                                                       remove_columns)
from ml.data.config.schemas.processed import ProcessedConfig
from ml.data.config.validate_config import validate_config
from ml.data.utils.memory.compute_memory_change import compute_memory_change
from ml.data.utils.memory.get_memory_usage import get_memory_usage
from ml.data.utils.persistence.save_data import save_data
from ml.logging_config import setup_logging
from ml.registry.row_id_registry import ROW_ID_REQUIRED
from ml.utils.data.get_data_suffix_and_format import get_data_suffix_and_format
from ml.utils.data.validate_data import validate_data
from ml.utils.formatting.iso_no_col import iso_no_colon
from ml.utils.loaders import load_json, load_yaml, read_data
from ml.utils.persistence.save_metadata import save_metadata
from ml.utils.snapshots.snapshot_path import get_snapshot_path

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make processed data by removing columns and creating new ones.")

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
        "--interim-snapshot-id",
        type=str,
        default="latest",
        help="Snapshot ID for the interim data version to use (optional; defaults to latest if not provided)"
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
    args: argparse.Namespace
    config_raw: dict
    config: ProcessedConfig
    df: pd.DataFrame
    data_path: Path
    metadata: dict

    args = parse_args()

    start_time = time.perf_counter()

    timestamp = iso_no_colon(datetime.now())
    processed_id = f"{timestamp}_{uuid4().hex[:8]}"
    
    data_dir = Path("data/processed") / args.data / args.version / processed_id
    data_dir.mkdir(parents=True, exist_ok=False)

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    setup_logging(data_dir / "make_processed.log", level=log_level)

    try:
        config_raw = load_yaml(Path(f"configs/data/processed/{args.data}/{args.version}.yaml"))
        config = validate_config(config_raw, type="processed")

        interim_data_dir = Path("data/interim") / args.data / config.interim_data_version

        interim_data_snapshot_path = get_snapshot_path(args.interim_snapshot_id, interim_data_dir)

        interim_metadata = load_json(interim_data_snapshot_path / "metadata.json")

        interim_data_suffix, interim_data_format = get_data_suffix_and_format(interim_metadata, location="data/output")

        interim_data_path = interim_data_snapshot_path / interim_data_suffix
        validate_data(data_path=interim_data_path, metadata=interim_metadata)
        logger.debug(f"Validated interim data at {interim_data_path} with metadata: {interim_metadata}")

        df = read_data(interim_data_format, interim_data_path)
        logger.debug(f"Read interim data from {interim_data_path} with shape {df.shape} and columns {df.columns.tolist()}.")

        df = remove_columns(df, config.remove_columns)
        logger.info(f"Removed columns {config.remove_columns}. Data shape is now {df.shape} with columns {df.columns.tolist()}.")

        row_id_info = None
        if config.data.name in ROW_ID_REQUIRED:
            df, row_id_info = add_row_id(df, config)
            logger.info(f"Added row_id to data. Data shape is now {df.shape} with columns {df.columns.tolist()}. Row ID info: {row_id_info}")

        data_path = save_data(df, config=config, data_dir=data_dir)
        logger.info(f"Processed data saved to {data_path}.")

        memory_usage = get_memory_usage(df)

        memory_info = compute_memory_change(target_metadata=interim_metadata, new_memory_usage=memory_usage, stage="processed")
        
        metadata = prepare_metadata(
            df, 
            config=config, 
            start_time=start_time, 
            data_path=data_path, 
            source_data_path=interim_data_path,
            source_data_format=interim_data_format,
            source_data_version=config.interim_data_version,
            owner=args.owner, 
            memory_info=memory_info, 
            processed_run_id=processed_id,
            row_id_info=row_id_info if config.data.name in ROW_ID_REQUIRED else None
        )

        save_metadata(metadata, target_dir=data_dir)

        return 0
    
    except Exception as e:
        exit_code = resolve_exit_code(e)
        return exit_code

if __name__ == "__main__":
    sys.exit(main())