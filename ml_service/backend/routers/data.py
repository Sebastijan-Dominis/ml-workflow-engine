"""API endpoints for editing interim and processed data configs."""

import copy
import os
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from ml.data.config.schemas.interim import InterimConfig
from ml.data.config.schemas.processed import ProcessedConfig

from ml_service.backend.configs.formatting.timestamp import add_timestamp
from ml_service.backend.main import limiter

router = APIRouter(prefix="/data", tags=["data"])

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root


def get_config_path(config_type: str, dataset_name: str, dataset_version: str) -> Path:
    """Return path for a given config type, dataset, and version.

    Args:
        config_type: "interim" or "processed"
        dataset_name: Name of the dataset
        dataset_version: Version of the dataset

    Returns:
        Path object pointing to the config file location
    """
    return Path(repo_root) / "configs" / "data" / config_type / dataset_name / f"{dataset_version}.yaml"


def validate_config_payload(config_type: str, payload: dict) -> InterimConfig | ProcessedConfig:
    """Validate payload with the correct schema.

    Args:
        config_type: "interim" or "processed"
        payload: Configuration payload to validate

    Returns:
        Validated config object
    """
    if config_type == "interim":
        return InterimConfig(**payload)
    elif config_type == "processed":
        return ProcessedConfig(**payload)
    else:
        raise ValueError(f"Unknown config_type: {config_type}")


def load_yaml_and_add_lineage(yaml_text: str) -> dict:
    """
    Parse YAML, ensure lineage exists, and inject timestamp.

    Args:
        yaml_text: YAML string payload

    Returns:
        dict: YAML parsed into dict with lineage.created_at
    """
    data = yaml.safe_load(yaml_text)

    data = add_timestamp(data, lineage_key="lineage")
    return data


@router.post("/validate", status_code=200)
@limiter.limit("1/15seconds")
def validate_yaml(payload: dict, request: Request):
    """Validate interim or processed config and check for collisions."""

    try:
        config_type = payload.get("type")
        yaml_text = payload.get("config")

        if not config_type or config_type not in ["interim", "processed"]:
            raise ValueError("Missing or invalid config type (must be 'interim' or 'processed').")
        if not yaml_text:
            raise ValueError("Missing config payload.")

        data_dict = load_yaml_and_add_lineage(yaml_text)

        dataset_name = data_dict.get("data", {}).get("name")
        dataset_version = data_dict.get("data", {}).get("version")

        if not dataset_name or not dataset_version:
            raise ValueError("Missing 'data.name' or 'data.version' in config YAML.")

        safe_dict = copy.deepcopy(data_dict)
        _ = validate_config_payload(config_type, data_dict)

        config_path = get_config_path(config_type, dataset_name, dataset_version)
        exists = config_path.exists()

        return {
            "valid": True,
            "exists": exists,
            "normalized": safe_dict, # Return the dict with lineage for frontend normalization
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}


@router.post("/write", status_code=201)
@limiter.limit("1/minute")
def write_yaml(payload: dict, request: Request):
    """Validate and write config atomically."""

    try:
        config_type = payload.get("type")
        yaml_text = payload.get("config")

        if not config_type or config_type not in ["interim", "processed"]:
            raise ValueError("Missing or invalid config type.")
        if not yaml_text:
            raise ValueError("Missing config payload.")

        data_dict = load_yaml_and_add_lineage(yaml_text)

        dataset_name = data_dict.get("data", {}).get("name")
        dataset_version = data_dict.get("data", {}).get("version")

        if not dataset_name or not dataset_version:
            raise ValueError("Missing 'data.name' or 'data.version' in config YAML.")

        safe_dict = copy.deepcopy(data_dict)
        _ = validate_config_payload(config_type, data_dict)

        config_path = get_config_path(config_type, dataset_name, dataset_version)

        if config_path.exists():
            return {
                "status": "exists",
                "message": f"{dataset_name}/{dataset_version} already exists for {config_type}",
            }

        config_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = config_path.parent / f"{config_path.name}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(safe_dict, f, sort_keys=False)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, config_path)

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()

            raise HTTPException(
                status_code=500,
                detail=f"Failed to write config: {str(e)}"
            ) from None

        return {"status": "written", "path": str(config_path)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
