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
from ml.feature_freezing.freeze_strategies.config.validate_feature_registry import validate_feature_registry
from ml.feature_freezing.utils.get_strategy import get_strategy
from ml.feature_freezing.persistence.save_metadata import save_metadata
from ml.logging_config import setup_logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Freeze features.")
    parser.add_argument("--problem", type=str, required=True, help="Problem name")
    parser.add_argument("--segment", type=str, required=True, help="Segment name")
    parser.add_argument("--feature_set", type=str, required=True, help="Feature set name")
    parser.add_argument("--version", type=str, required=True, help="Feature set version")
    parser.add_argument("--data_type", type=str, required=True, help="Data type (e.g. tabular, time_series)")
    parser.add_argument("--logging-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)")
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

    # Generate the snapshot id once so it can be reused for both the
    # log destination and the data persistence step.
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    snapshot_id = f"{timestamp}_{uuid4().hex[:8]}"

    try:
        config_raw = load_feature_registry(args.problem, args.segment, args.feature_set, args.version)
        config = validate_feature_registry(config_raw, args.data_type)

        # Now that the config (and its feature_store_path) is available
        # set up logging inside the snapshot directory.
        log_path = Path(config.feature_store_path) / snapshot_id / "freeze.log"
        setup_logging(log_path, level=log_level)

        if config.type != args.data_type:
            msg = f"Data type mismatch: expected {args.data_type}, got {config.type}"
            logger.error(msg)
            raise UserError(msg)

        strategy = get_strategy(config.type)

        snapshot_path, metadata = strategy.freeze(config, snapshot_id=snapshot_id, timestamp=timestamp, start_time=start_time)
        save_metadata(snapshot_path, metadata)

        return 0

    except Exception as e:
        logger.exception("Feature freezing failed")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())


