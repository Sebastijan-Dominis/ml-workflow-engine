import logging
logger = logging.getLogger(__name__)
import sys
import argparse
from pathlib import Path
from typing import Any, Protocol
from pydantic import ValidationError

from ml.search.logging_config import setup_logging
from ml.utils.utils import load_and_validate_config
from ml.search.persistence.save_experiment import save_experiment
from ml.registry.search_registry import SEARCH_REGISTRY

class Searcher(Protocol):
    """
    Searcher interface.

    Returns:
        dict with keys:
        - best_params
        - phases
    """
    def search(self, model_cfg: dict[str, Any]) -> dict[str, Any]: ...

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
        "--owner",
        type=str,
        default="Sebastijan",
        help="Owner of the experiment (default: Sebastijan)"
    )

    return parser.parse_args()

def get_searcher(model_cfg: dict[str, Any]) -> Searcher:
        key = model_cfg["algorithm"].value.lower()

        searcher_cls = SEARCH_REGISTRY.get(key)

        if not searcher_cls:
            msg = f"No searcher registered for algorithm {model_cfg['algorithm']}."
            logger.error(msg)
            raise ValueError(msg)

        searcher = searcher_cls()

        logger.info(
            "Using searcher %s for algorithm=%s",
            searcher_cls.__name__,
            model_cfg["algorithm"].value,
        )

        return searcher

def main() -> int:
    """Main function to perform hyperparameter search and save training configuration."""
    args: argparse.Namespace
    model_cfg: dict[str, Any]
    searcher: Searcher
    search_results: dict[str, Any]
    
    setup_logging()

    try:
        args = parse_args()

        model_cfg = load_and_validate_config(Path(f"configs/search/{args.problem}/{args.segment}/{args.version}.yaml"), cfg_type="search", env=args.env)

        logger.info("Using config: %s, environment: %s", model_cfg["_meta"].get("sources", {}).get("main"), model_cfg["_meta"].get("env"))

        searcher = get_searcher(model_cfg)

        search_results = searcher.search(model_cfg)

        save_experiment(model_cfg, search_results, args.owner)

        logger.info(
            "Search completed | problem=%s segment=%s version=%s",
            args.problem,
            args.segment,
            args.version,
        )

        return 0
    
    except (ValidationError, FileNotFoundError) as e:
        logger.error("Configuration error: %s", e)
        logger.error("Fix the configuration and try again.")
        return 2

    except Exception as e:
        logger.exception("An error occurred during hyperparameter search or configuration saving.")
        return 1

if __name__ == "__main__":
    sys.exit(main())