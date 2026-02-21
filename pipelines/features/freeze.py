import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml

from ml.cli.error_handling import resolve_exit_code
from ml.exceptions import UserError
from ml.feature_freezing.freeze_strategies.config.validate_feature_registry import \
    validate_feature_registry
from ml.feature_freezing.utils.get_strategy import get_strategy
from ml.logging_config import add_file_handler, bootstrap_logging
from ml.utils.persistence.save_metadata import save_metadata

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Freeze features.")

    parser.add_argument(
        "--problem", 
        type=str, 
        required=True, 
        help="Model problem, e.g., 'no_show'"
    )

    parser.add_argument(
        "--segment", 
        type=str, 
        required=True, 
        help="Model segment name, e.g., 'city_hotel_online_ta'"
    )

    parser.add_argument(
        "--feature_set", 
        type=str, 
        required=True, 
        help="Feature set name, e.g., 'base_features'"
    )

    parser.add_argument(
        "--version", 
        type=str, 
        required=True, 
        help="Feature set version, e.g., 'v1'"
    )

    parser.add_argument(
        "--data_type", 
        choices=["tabular", "time_series"], 
        required=True, 
        help="Data type (tabular or time_series)"
    )

    parser.add_argument(
        "--logging-level", 
        type=str, 
        default="INFO", 
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    return parser.parse_args()

def load_feature_registry(problem, segment, feature_set, version) -> dict:
    path = Path(f"configs/feature_registry/features.yaml")
    with open(path, "r") as f:
        registry = yaml.safe_load(f)
    return registry[problem][segment][feature_set][version]

def main() -> int:
    args = parse_args()

    start_time = time.perf_counter()

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    bootstrap_logging(level=log_level)

    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    snapshot_id = f"{timestamp}_{uuid4().hex[:8]}"

    try:
        config_raw = load_feature_registry(args.problem, args.segment, args.feature_set, args.version)
        config = validate_feature_registry(config_raw, args.data_type)

        log_path = Path(config.feature_store_path) / snapshot_id / "freeze.log"
        add_file_handler(log_path, level=log_level)

        if config.type != args.data_type:
            msg = f"Data type mismatch: expected {args.data_type}, got {config.type}"
            logger.error(msg)
            raise UserError(msg)

        strategy = get_strategy(config.type)

        snapshot_path, metadata = strategy.freeze(config, snapshot_id=snapshot_id, timestamp=timestamp, start_time=start_time)
        save_metadata(metadata, target_dir=snapshot_path)

        return 0

    except Exception as e:
        logger.exception("Feature freezing failed")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())
