"""CLI for freezing and versioning feature sets.

The script loads feature registry configuration, resolves the appropriate freeze
strategy, materializes a feature snapshot, and writes snapshot metadata.
"""

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
from ml.feature_freezing.freeze_strategies.config.validate_feature_registry import (
    validate_feature_registry,
)
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.feature_freezing.utils.get_strategy import get_strategy
from ml.feature_freezing.utils.get_strategy_type import get_strategy_type
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.persistence.save_metadata import save_metadata
from ml.logging_config import add_file_handler, bootstrap_logging

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature freezing.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
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
        "--snapshot-binding-key",
        type=str,
        help="Optional key for a snapshot binding to define which snapshot to load for each dataset. Snapshots should be defined in configs/snapshot_bindings_registry/bindings.yaml. Example value: '2026-03-20T02-54-47_61509023'",
        default=None
    )

    parser.add_argument(
        "--owner",
        type=str,
        default="Sebastijan",
        help="Owner of the feature set (default: Sebastijan)"
    )

    parser.add_argument(
        "--logging-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    return parser.parse_args()

def load_feature_registry(feature_set: str, version: str) -> dict:
    """Load feature registry configuration for a feature set version.

    Args:
        feature_set: Name of the feature set in the registry.
        version: Version key under the feature set.

    Returns:
        dict: Raw registry configuration for the requested feature set version.
    """
    path = Path("configs/feature_registry/features.yaml")
    with open(path, encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    return dict(registry[feature_set][version])

def main() -> int:
    """Execute the feature freeze workflow.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Exceptions are converted to process exit codes; the function is designed
        as a CLI boundary and does not propagate failures upward.

    Side Effects:
        Creates a feature snapshot directory, writes freeze logs, and persists
        feature snapshot metadata.

    Examples:
        python pipelines/features/freeze.py --feature-set booking_context_features --version v1
    """
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

    timestamp = iso_no_colon(datetime.now())
    snapshot_id = f"{timestamp}_{uuid4().hex[:8]}"

    try:
        config_raw = load_feature_registry(args.feature_set, args.version)
        strategy_type = get_strategy_type(config_raw)
        config = validate_feature_registry(config_raw, strategy_type)
    except Exception as e:
        logging.exception("Failed to load and validate feature registry")
        return resolve_exit_code(e)

    log_path = Path(config.feature_store_path) / snapshot_id / "freeze.log"
    add_file_handler(log_path, level=log_level)

    try:

        strategy = get_strategy(config.type)

        output = strategy.freeze(
            config,
            snapshot_binding_key=args.snapshot_binding_key,
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            start_time=start_time,
            owner=args.owner
        )
        snapshot_path = output.snapshot_path
        metadata = output.metadata

        save_metadata(metadata, target_dir=snapshot_path)

        return 0

    except Exception as e:
        logger.exception("Feature freezing failed")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())
