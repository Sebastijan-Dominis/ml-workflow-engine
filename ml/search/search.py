import logging
logger = logging.getLogger(__name__)
import sys
import argparse
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from typing import Any, Protocol

from ml.logging_config import setup_logging
from ml.config.loader import load_and_validate_config
from ml.search.persistence.save_experiment import save_experiment
from ml.registry.search_registry import SEARCH_REGISTRY
from ml.cli.error_handling import resolve_exit_code
from ml.exceptions import UserError, PipelineContractError
from ml.config.validation_schemas.model_cfg import SearchModelConfig

class Searcher(Protocol):
    """
    Searcher interface.

    Returns:
        dict with keys:
        - best_params
        - phases
    """
    def search(self, model_cfg: SearchModelConfig) -> dict[str, Any]: ...

def parse_args() -> argparse.Namespace:
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
        "--env",
        type=str,
        default="default",
        help="Environment to run the script in (dev/test/prod) (default: default) ~ none"
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

def get_searcher(model_cfg: SearchModelConfig) -> Searcher:
        key = model_cfg.algorithm.value.lower()

        searcher_cls = SEARCH_REGISTRY.get(key)

        if not searcher_cls:
            msg = f"No searcher registered for algorithm {model_cfg.algorithm}."
            logger.error(msg)
            raise PipelineContractError(msg)

        searcher = searcher_cls()

        logger.info(
            "Using searcher %s for algorithm=%s",
            searcher_cls.__name__,
            model_cfg.algorithm.value,
        )

        return searcher

def main() -> int:
    """Main function to perform hyperparameter search and save training configuration."""
    args: argparse.Namespace
    model_cfg: SearchModelConfig 
    searcher: Searcher
    search_results: dict[str, Any]
    
    args = parse_args()

    # Generate experiment id up-front so the log file can be placed
    # inside the experiment directory before any work starts.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{timestamp}_{uuid4().hex[:8]}"
    experiment_dir = Path("experiments") / args.problem / args.segment / args.version / experiment_id

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    setup_logging(experiment_dir / "search.log", level=log_level)

    try:
        model_cfg = load_and_validate_config(Path(f"configs/search/{args.problem}/{args.segment}/{args.version}.yaml"), cfg_type="search", env=args.env)

        logger.info("Using config: %s, environment: %s", model_cfg.meta.sources.get("main") if model_cfg.meta.sources else None, model_cfg.meta.env)

        searcher = get_searcher(model_cfg)

        search_results = searcher.search(model_cfg)

        save_experiment(model_cfg, search_results, args.owner, experiment_id=experiment_id)

        logger.info(
            "Search completed | problem=%s segment=%s version=%s experiment_id=%s",
            args.problem,
            args.segment,
            args.version,
            experiment_id,
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