import os
from dataclasses import dataclass
from datetime import UTC, datetime

import yaml
from fastapi import APIRouter, HTTPException, Request
from ml.config.schemas.model_specs import ModelSpecs
from ml.config.schemas.search_cfg import SearchConfig
from ml.config.schemas.train_cfg import TrainConfig
from ml_service.backend.limiter import limiter
from pydantic import BaseModel, Field


class SearchConfigForValidation(BaseModel):
    """Separate schema for search config validation to allow extra fields for lineage tracking and extension."""
    extends: list[str] = Field(default_factory=list)
    search_lineage: dict
    search: SearchConfig

class TrainConfigForValidation(BaseModel):
    """Separate schema for train config validation to allow extra fields for lineage tracking and extension."""
    extends: list[str] = Field(default_factory=list)
    training_lineage: dict
    training: TrainConfig

@dataclass
class RawConfigsWithLineage:
    model_specs: dict
    search: dict
    training: dict

@dataclass
class ValidatedConfigs:
    model_specs: ModelSpecs
    search: SearchConfigForValidation
    training: TrainConfigForValidation

router = APIRouter(prefix="/modeling", tags=["modeling"])

@dataclass
class ConfigPaths:
    model_specs: str
    search: str
    training: str


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root

def utc_timestamp():
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )

def add_timestamp(data: dict, lineage_key: str) -> dict:
    if lineage_key not in data:
        raise ValueError(f"Missing '{lineage_key}' key in config data for lineage tracking.")
    data[lineage_key]["created_at"] = utc_timestamp()
    return data

def load_all_yamls_and_add_lineage(payload: dict) -> RawConfigsWithLineage:
    try:
        model_specs_yaml = payload.get("model_specs", "{}")
        search_yaml = payload.get("search", "{}")
        training_yaml = payload.get("training", "{}")

        model_specs_data = yaml.safe_load(model_specs_yaml) or {}
        search_data = yaml.safe_load(search_yaml) or {}
        training_data = yaml.safe_load(training_yaml) or {}

        if not model_specs_data or not search_data or not training_data:
            raise ValueError("One or more configs are empty or invalid YAML.")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parsing error: {str(e)}") from e

    model_specs_with_lineage = add_timestamp(model_specs_data, "model_specs_lineage")
    search_with_lineage = add_timestamp(search_data, "search_lineage")
    training_with_lineage = add_timestamp(training_data, "training_lineage")

    return RawConfigsWithLineage(
        model_specs=model_specs_with_lineage,
        search=search_with_lineage,
        training=training_with_lineage
    )

def validate_all_configs(data_with_lineage: RawConfigsWithLineage) -> ValidatedConfigs:
    try:
        model_specs = ModelSpecs(**data_with_lineage.model_specs)
        search = SearchConfigForValidation(**data_with_lineage.search)
        training = TrainConfigForValidation(**data_with_lineage.training)
        return ValidatedConfigs(model_specs=model_specs, search=search, training=training)
    except Exception as e:
        raise ValueError(f"Config validation error: {str(e)}") from e

def compute_paths(validated_configs: ValidatedConfigs) -> ConfigPaths:
    model_specs = validated_configs.model_specs

    model_specs_path = f"{repo_root}/configs/model_specs/{model_specs.problem}/{model_specs.segment.name}/{model_specs.version}.yaml"
    search_path = f"{repo_root}/configs/search/{model_specs.problem}/{model_specs.segment.name}/{model_specs.version}.yaml"
    training_path = f"{repo_root}/configs/train/{model_specs.problem}/{model_specs.segment.name}/{model_specs.version}.yaml"

    return ConfigPaths(
        model_specs=model_specs_path,
        search=search_path,
        training=training_path
    )

def check_paths(validated_configs: ValidatedConfigs) -> ConfigPaths:
    paths = compute_paths(validated_configs)

    if os.path.exists(paths.model_specs):
        raise FileExistsError(f"Model specs config already exists at {paths.model_specs}\nOverwriting existing configs is not allowed.")
    if os.path.exists(paths.search):
        raise FileExistsError(f"Search config already exists at {paths.search}\nOverwriting existing configs is not allowed.")
    if os.path.exists(paths.training):
        raise FileExistsError(f"Training config already exists at {paths.training}\nOverwriting existing configs is not allowed.")

    return paths

def save_all_configs(validated_configs: ValidatedConfigs, paths: ConfigPaths) -> None:
    os.makedirs(os.path.dirname(paths.model_specs), exist_ok=True)
    os.makedirs(os.path.dirname(paths.search), exist_ok=True)
    os.makedirs(os.path.dirname(paths.training), exist_ok=True)

    with open(paths.model_specs, "w") as f:
        yaml.safe_dump(
            validated_configs.model_specs.model_dump(mode="json", exclude={"meta"}),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
    with open(paths.search, "w") as f:
        yaml.safe_dump(
            validated_configs.search.model_dump(mode="json"),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
    with open(paths.training, "w") as f:
        yaml.safe_dump(
            validated_configs.training.model_dump(mode="json"),
            f,
            sort_keys=False,
            default_flow_style=False,
        )

@router.post("/validate", status_code=200)
@limiter.limit("3/minute")
def validate_yaml(payload: dict, request: Request):
    try:
        data_with_lineage = load_all_yamls_and_add_lineage(payload)

        validated_configs = validate_all_configs(data_with_lineage)

        check_paths(validated_configs)

        return {
            "valid": True,
            "normalized": {
                "model_specs": validated_configs.model_specs.model_dump(mode="json", exclude={"meta"}) ,
                "search": validated_configs.search.model_dump(mode="json"),
                "training": validated_configs.training.model_dump(mode="json"),
            }
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


@router.post("/write", status_code=201)
@limiter.limit("1/minute")
def write_yaml(payload: dict, request: Request):
    try:
        data_with_lineage = load_all_yamls_and_add_lineage(payload)

        validated_configs = validate_all_configs(data_with_lineage)

        paths = check_paths(validated_configs)

        save_all_configs(validated_configs, paths)

        return {
            "paths": {
                    "model_specs": paths.model_specs,
                    "search": paths.search,
                    "training": paths.training
                }
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
