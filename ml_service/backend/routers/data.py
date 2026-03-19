"""API endpoints for editing interim and processed data configs."""

import copy
import os

from fastapi import APIRouter, HTTPException, Request

from ml_service.backend.configs.data.utils.get_config_path import get_config_path
from ml_service.backend.configs.data.validation.validate_config_payload import (
    validate_config_payload,
)
from ml_service.backend.configs.loading.load_yaml_and_add_lineage import load_yaml_and_add_lineage
from ml_service.backend.configs.persistence.save_config import save_config
from ml_service.backend.main import limiter

router = APIRouter(prefix="/data", tags=["data"])

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root

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

        config_path = get_config_path(
            repo_root=repo_root,
            config_type=config_type,
            dataset_name=dataset_name,
            dataset_version=dataset_version
        )
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

        config_path = get_config_path(
            repo_root=repo_root,
            config_type=config_type,
            dataset_name=dataset_name,
            dataset_version=dataset_version
        )

        if config_path.exists():
            return {
                "status": "exists",
                "message": f"{dataset_name}/{dataset_version} already exists for {config_type}",
            }

        save_config(safe_dict, config_path)

        return {"status": "written", "path": str(config_path)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
