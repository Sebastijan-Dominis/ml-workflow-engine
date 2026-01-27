import logging

import yaml
logger = logging.getLogger(__name__)
import argparse

from pathlib import Path
from ml.search.logging_config import setup_logging
from ml.search.persistence.save_training_config import save_training_config
from ml.search.custom_search_scripts.catboost_classification_v1 import search_catboost_classification_v1

def parse_args() -> argparse.Namespace:
    try:
        parser = argparse.ArgumentParser(description="Search for best hyperparameters and save training configuration.")

        parser.add_argument(
            "--name",
            type=str,
            required=True,
            help="Model name, e.g., 'no_show'"
        )

        parser.add_argument(
            "--segment",
            type=str,
            required=True,
            help="Model segment, e.g., 'city_hotel_online_ta'"
        )

        parser.add_argument(
            "--version",
            type=str,
            required=True,
            help="Model version, e.g., 'v1'"
        )

        return parser.parse_args()
    
    except Exception:
        logger.exception("Failed to parse arguments.")
        raise

def load_config(name, segment, version) -> dict:
    config_path = Path(f"ml/search/search_configs/{name}/{segment}/{version}.yaml")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception:
        logger.exception(f"Failed to load configuration from {config_path}.")
        raise

def main() -> None:
    """Main function to perform hyperparameter search and save training configuration."""
    
    setup_logging()

    args = parse_args()

    cfg = load_config(args.name, args.segment, args.version)

    # Searcher registry: extend this dict when adding new search scripts
    SEARCHERS = {
        "catboost_classification_v1": search_catboost_classification_v1,
    }

    searcher = SEARCHERS.get(cfg["search"]["search_script"])

    if searcher:
        best_params = searcher(cfg)
    else:
        logger.error(f"Search script {cfg['search_script']} not found in registry.")
        raise ValueError(f"Search script {cfg['search_script']} not found.")

    save_training_config(cfg, best_params)

    logger.info(f"Hyperparameter search and configuration saving for model {args.name}_{args.segment}_{args.version} completed successfully.")

if __name__ == "__main__":
    main()