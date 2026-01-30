import yaml
import sys
import pandas as pd
from pathlib import Path
from pydantic_core import ValidationError

from ml.validation_schemas.model_specs import ModelSpecsSchema

def load_model_specs(problem, segment, version, logger) -> dict:
    config_path = Path(f"configs/model_specs/{problem}/{segment}/{version}.yaml")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception:
        logger.exception(f"Failed to load configuration from {config_path}.")
        raise

def validate_model_specs(cfg_raw: dict, logger) -> dict:
    try:
        cfg = ModelSpecsSchema(**cfg_raw).model_dump()
        return cfg
    except ValidationError as e:
        logger.error("Config validation failed:")
        for err in e.errors():
            logger.error("Field %s: %s", ".".join(map(str, err['loc'])), err['msg'])
        sys.exit(1)  # Stop execution if config is invalid

def get_cat_features(raw_schema: pd.DataFrame, derived_schema: pd.DataFrame) -> list:
    raw_categoricals = raw_schema.loc[
        raw_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    derived_categoricals = derived_schema.loc[
        derived_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    return raw_categoricals + derived_categoricals

def load_schemas(features_path: Path, logger) -> tuple:
    raw_schema_path = features_path / "schema.csv"
    derived_schema_path = features_path / "derived_schema.csv"
    try:
        raw_schema = pd.read_csv(raw_schema_path)
        derived_schema = pd.read_csv(derived_schema_path)
        return raw_schema, derived_schema
    except Exception:
        logger.exception(f"Failed to load schemas from {features_path}.")
        raise
    