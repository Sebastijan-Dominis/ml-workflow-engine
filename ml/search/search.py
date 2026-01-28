import logging

import yaml
logger = logging.getLogger(__name__)
import argparse
import pandas as pd

from pathlib import Path
from ml.search.logging_config import setup_logging
from ml.search.persistence.save_training_config import save_training_config
from ml.search.custom_search_scripts.catboost import search_catboost

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

def load_model_specs(name, segment, version) -> dict:
    config_path = Path(f"configs/model_specs/{name}/{segment}/{version}.yaml")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception:
        logger.exception(f"Failed to load configuration from {config_path}.")
        raise

def load_search_configs(version) -> dict:
    config_path = Path(f"configs/search/{version}.yaml")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception:
        logger.exception(f"Failed to load search configuration from {config_path}.")
        raise

def main() -> None:
    """Main function to perform hyperparameter search and save training configuration."""
    
    setup_logging()

    args = parse_args()

    cfg_model_specs = load_model_specs(args.name, args.segment, args.version)

    cfg_search = load_search_configs(cfg_model_specs["search_version"])

    #Searcher registry: extend this dict when adding new search scripts
    SEARCHERS = {
        "catboost": search_catboost,
    }

    searcher = SEARCHERS.get(cfg_search["search_script"])

    if searcher:
        best_params = searcher(cfg_model_specs, cfg_search)
    else:
        logger.error(f"Search script {cfg_search['search_script']} not found in registry.")
        raise ValueError(f"Search script {cfg_search['search_script']} not found.")

    # save_training_config(cfg_model_specs, best_params)

    logger.info(f"Best hyperparameters found: {best_params}")

    logger.info(f"Hyperparameter search and configuration saving for model {args.name}_{args.segment}_{args.version} completed successfully.")

if __name__ == "__main__":
    main()