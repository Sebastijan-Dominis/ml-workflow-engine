import logging
logger = logging.getLogger(__name__)

import yaml
import subprocess
from pathlib import Path
from pydantic_core import ValidationError

from ml.validation_schemas.model_specs import ModelSpecsSchema

def load_model_specs(problem, segment, version) -> dict:
    config_path = Path(f"configs/model_specs/{problem}/{segment}/{version}.yaml")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception:
        logger.exception(f"Failed to load configuration from {config_path}.")
        raise

def validate_model_specs(cfg_raw: dict) -> dict:
    try:
        cfg = ModelSpecsSchema(**cfg_raw).model_dump()
        return cfg
    except ValidationError as e:
        logger.error("Config validation failed:")
        for err in e.errors():
            logger.error("Field %s: %s", ".".join(map(str, err['loc'])), err['msg'])
        raise

def get_git_commit(repo_dir: Path = Path(".")) -> str:
    try:
        # Find the top-level git directory
        top_level = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        # Get the HEAD commit hash
        commit_hash = subprocess.check_output(
            ["git", "-C", top_level, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        return commit_hash
    except subprocess.CalledProcessError:
        return "unknown"
