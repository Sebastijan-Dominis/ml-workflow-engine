"""Hyperparameter search runner CLI.

This module executes model search for a given problem/segment/version,
persists search outputs, and manages failure-management cleanup.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ml.cli.error_handling import resolve_exit_code
from ml.config.hashing import add_config_hash
from ml.config.loader import load_and_validate_config
from ml.config.schemas.model_cfg import SearchModelConfig
from ml.exceptions import UserError
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.formatting.str_to_bool import str_to_bool
from ml.logging_config import setup_logging
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.search.persistence.persist_experiment import persist_experiment
from ml.search.searchers.base import Searcher
from ml.search.searchers.output import SearchOutput
from ml.search.utils.failure_management.delete_failure_management_folder import (
    delete_failure_management_folder,
)
from ml.search.utils.get_searcher import get_searcher

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hyperparameter search.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Search for best hyperparameters and save training configuration.")

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
        "--version",
        type=str,
        required=True,
        help="Model version, e.g., 'v1'"
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID to use for this run (default: None, which generates a new unique experiment ID). If provided, it should be in the format 'timestamp_randomstring', e.g., '20240101T120000_abcdef12'."
    )

    parser.add_argument(
        "--env",
        choices=["dev", "test", "prod", "default"],
        default="default",
        help="Environment to run the script in (dev/test/prod) (default: default) ~ none"
    )

    parser.add_argument(
        "--strict",
        type=str_to_bool,
        default=True,
        help="Whether to run in strict mode, which includes strict validation that may be computationally expensive (default: True)"
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
        help="Owner of the experiment (default: Sebastijan)"
    )

    parser.add_argument(
        "--clean-up-failure-management",
        type=str_to_bool,
        default=True,
        help="Whether to clean up failure management folder after successful run (default: True)"
    )

    parser.add_argument(
        "--overwrite-existing",
        type=str_to_bool,
        default=False,
        help="Whether to overwrite existing experiment data if the experiment ID already exists (default: False). If False and files (other than search.log) already exist within the experiment, the script will raise an error to prevent accidental data loss."
    )

    return parser.parse_args()

def main() -> int:
    """Perform hyperparameter search and persist experiment artifacts.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Exceptions are converted to process exit codes; the function is designed
        as a CLI boundary and does not propagate failures upward.

    Side Effects:
        Creates/updates search run directories, writes logs/metadata/runtime
        artifacts, and may delete failure-management folders on success.

    Examples:
        python pipelines/search/search.py --problem cancellation --segment global --version v1
    """
    args: argparse.Namespace
    model_cfg: SearchModelConfig
    searcher: Searcher
    search_results: dict[str, Any]
    feature_lineage: list[FeatureLineage]
    pipeline_hash: str
    start_time: float
    search_output: SearchOutput

    args = parse_args()

    start_time = time.perf_counter()

    timestamp = iso_no_colon(datetime.now())
    experiment_id = args.experiment_id if args.experiment_id else f"{timestamp}_{uuid4().hex[:8]}"
    experiment_dir = Path("experiments") / args.problem / args.segment / args.version / experiment_id

    if not experiment_dir.exists() and args.experiment_id:
        msg = f"Experiment directory {experiment_dir} does not exist for provided experiment ID {experiment_id}."
        logger.error(msg)
        return 1

    search_dir = experiment_dir / "search"
    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)

    if search_dir.exists():
        setup_logging(search_dir / "search.log", level=log_level)
        existing_files = [f.name for f in search_dir.iterdir() if f.is_file() and f.name != "search.log"]
        if existing_files and not args.overwrite_existing:
            msg = f"Search directory {search_dir} already exists and contains files: {', '.join(existing_files)}. To prevent accidental data loss, the script will not overwrite existing experiment data. To run search with this experiment ID, please delete the existing files (if deemed appropriate), or set --overwrite-existing to True (if you want to overwrite the existing data)."
            logger.error(msg)
            return 1
    else:
        search_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(search_dir / "search.log", level=log_level)

    failure_management_dir = Path("failure_management") / experiment_id
    failure_management_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_cfg = load_and_validate_config(Path(f"configs/search/{args.problem}/{args.segment}/{args.version}.yaml"), cfg_type="search", env=args.env)

        model_cfg = add_config_hash(model_cfg)

        key = model_cfg.algorithm.value.lower()

        searcher = get_searcher(key)

        logger.info(f"Starting hyperparameter search for experiment {experiment_id}.")
        search_output = searcher.search(
            model_cfg,
            strict=args.strict,
            failure_management_dir=failure_management_dir,
        )
        logger.info("Search completed. Persisting search run...")

        search_results = search_output.search_results
        feature_lineage = search_output.feature_lineage
        pipeline_hash = search_output.pipeline_hash
        scoring_method = search_output.scoring_method
        splits_info = search_output.splits_info

        persist_experiment(
            model_cfg,
            search_results=search_results,
            owner=args.owner,
            experiment_id=experiment_id,
            search_dir=search_dir,
            timestamp=timestamp,
            start_time=start_time,
            feature_lineage=feature_lineage,
            pipeline_hash=pipeline_hash,
            scoring_method=scoring_method,
            splits_info=splits_info,
            overwrite_existing=args.overwrite_existing
        )

        logger.info("Search run successfully persisted.")

        delete_failure_management_folder(
            folder_path=failure_management_dir,
            cleanup=args.clean_up_failure_management,
            stage="search"
        )

        return 0

    except Exception as e:
        exit_code = resolve_exit_code(e)

        if isinstance(e, UserError):
            logger.error("%s", e)
        else:
            logger.exception(
                "An error occurred during hyperparameter search or configuration saving."
            )

        return exit_code

if __name__ == "__main__":
    sys.exit(main())
