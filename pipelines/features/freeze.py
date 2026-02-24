import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml

from ml.cli.error_handling import resolve_exit_code
from ml.feature_freezing.constants.output import FreezeOutput
from ml.feature_freezing.freeze_strategies.base import FreezeStrategy
from ml.feature_freezing.freeze_strategies.config.validate_feature_registry import \
    validate_feature_registry
from ml.feature_freezing.freeze_strategies.tabular.config.models import \
    TabularFeaturesConfig
from ml.feature_freezing.utils.get_strategy import get_strategy
from ml.feature_freezing.utils.get_strategy_type import get_strategy_type
from ml.logging_config import add_file_handler, bootstrap_logging
from ml.utils.persistence.save_metadata import save_metadata

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Freeze features.")

    parser.add_argument(
        "--feature-set", 
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
        "--logging-level", 
        type=str, 
        default="INFO", 
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    return parser.parse_args()

def load_feature_registry(feature_set: str, version: str) -> dict:
    path = Path(f"configs/feature_registry/features.yaml")
    with open(path, "r") as f:
        registry = yaml.safe_load(f)
    return registry[feature_set][version]

def main() -> int:
    args: argparse.Namespace
    start_time: float
    config_raw: dict
    strategy_type: str
    config: TabularFeaturesConfig
    strategy: FreezeStrategy
    output: FreezeOutput

    args = parse_args()

    start_time = time.perf_counter()

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    bootstrap_logging(level=log_level)

    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    snapshot_id = f"{timestamp}_{uuid4().hex[:8]}"

    try:
        config_raw = load_feature_registry(args.feature_set, args.version)
        strategy_type = get_strategy_type(config_raw)
        config = validate_feature_registry(config_raw, strategy_type)

        log_path = Path(config.feature_store_path) / snapshot_id / "freeze.log"
        add_file_handler(log_path, level=log_level)

        strategy = get_strategy(config.type)

        output = strategy.freeze(config, snapshot_id=snapshot_id, timestamp=timestamp, start_time=start_time)
        snapshot_path = output.snapshot_path
        metadata = output.metadata

        save_metadata(metadata, target_dir=snapshot_path)

        return 0

    except Exception as e:
        logger.exception("Feature freezing failed")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())
