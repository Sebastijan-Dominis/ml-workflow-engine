import logging
logger = logging.getLogger(__name__)
import sys
import yaml
import argparse
from pydantic_core import ValidationError
from pathlib import Path

from ml.search.logging_config import setup_logging
from ml.utils import load_model_specs, validate_model_specs
from ml.validation_schemas.search import SearchSchemaV1
from ml.search.custom_search_scripts.catboost import search_catboost
from ml.search.persistence.save_training_config import save_training_config

def parse_args() -> argparse.Namespace:
    try:
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
    
    except Exception:
        logger.exception("Failed to parse arguments.")
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

def validate_search_configs(cfg_raw, version) -> dict:
    SEARCH_SCHEMAS = {
        "v1": SearchSchemaV1,
    }

    try:
        SchemaClass = SEARCH_SCHEMAS.get(version)
        if not SchemaClass:
            raise ValueError(f"Unsupported search config version: {version}")
        cfg = SchemaClass(**cfg_raw).model_dump()
        return cfg
    except ValidationError as e:
        logger.error("Search config validation failed:")
        for err in e.errors():
            logger.error("Field %s: %s", ".".join(map(str, err['loc'])), err['msg'])
        sys.exit(1)  # Stop execution if config is invalid

def main() -> None:
    """Main function to perform hyperparameter search and save training configuration."""
    
    setup_logging()

    args = parse_args()

    cfg_model_specs_raw = load_model_specs(args.problem, args.segment, args.version, logger)

    cfg_model_specs = validate_model_specs(cfg_model_specs_raw, logger)

    cfg_search_raw = load_search_configs(cfg_model_specs["search_version"])

    cfg_search = validate_search_configs(cfg_search_raw, cfg_model_specs["search_version"])

    #Searcher registry: extend this dict when adding new search scripts
    SEARCHERS = {
        "catboost": search_catboost,
    }

    key = cfg_model_specs["algorithm"].lower()
    searcher = SEARCHERS.get(key)

    if searcher:
        best_params = searcher(cfg_model_specs, cfg_search)
    else:
        logger.error(f"Search script {cfg_search['search_script']} not found in registry.")
        raise ValueError(f"Search script {cfg_search['search_script']} not found in registry.")

    save_training_config(cfg_model_specs, best_params)

    logger.info(f"Hyperparameter search and configuration saving for model {args.problem}_{args.segment}_{args.version} completed successfully.")

if __name__ == "__main__":
    main()