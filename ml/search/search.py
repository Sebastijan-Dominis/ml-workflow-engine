import logging

logger = logging.getLogger(__name__)
import sys
import yaml
import argparse
from pydantic_core import ValidationError
from pathlib import Path

from ml.search.logging_config import setup_logging
from ml.utils import load_model_specs, validate_model_specs
from ml.validation_schemas.search import SearchConfig
from ml.search.persistence.save_experiment import save_experiment
from ml.search.searchers.catboost import SearchCatboost

SEARCH_REGISTRY = {
    "catboost": SearchCatboost,
}

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

    return parser.parse_args()

def load_search_configs(problem, segment, version) -> dict:
    config_path = Path(f"configs/search/{problem}/{segment}/{version}.yaml")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception:
        logger.exception(f"Failed to load search configuration from {config_path}.")
        raise

def validate_search_config(cfg_raw: dict) -> dict:
    try:
        cfg = SearchConfig(**cfg_raw).model_dump()
        return cfg
    except ValidationError as e:
        logger.error("Config validation failed:")
        for err in e.errors():
            logger.error("Field %s: %s", ".".join(map(str, err['loc'])), err['msg'])
        raise

def main() -> int:
    """Main function to perform hyperparameter search and save training configuration."""
    
    setup_logging()

    try:
        args = parse_args()

        model_specs_raw = load_model_specs(args.problem, args.segment, args.version)

        model_specs = validate_model_specs(model_specs_raw)

        search_configs_raw = load_search_configs(args.problem, args.segment, args.version)

        search_configs = validate_search_config(search_configs_raw)

        key = model_specs["algorithm"].lower()

        searcher_cls = SEARCH_REGISTRY.get(key)

        if not searcher_cls:
            msg = f"No searcher registered for algorithm {model_specs['algorithm']}."
            logger.error(msg)
            raise ValueError(msg)

        searcher = searcher_cls()

        logger.info(
            "Using searcher %s for algorithm=%s",
            searcher_cls.__name__,
            model_specs["algorithm"],
        )


        search_results = searcher.search(model_specs, search_configs)

        save_experiment(model_specs, search_configs, search_results)

        logger.info(
            "Search completed | problem=%s segment=%s version=%s",
            args.problem,
            args.segment,
            args.version,
        )

        return 0
    
    except Exception as e:
        logger.exception("An error occurred during hyperparameter search or configuration saving.")
        return 1

if __name__ == "__main__":
    sys.exit(main())