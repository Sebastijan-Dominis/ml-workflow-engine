"""API endpoints for feature registry editing."""
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ml_service.backend.configs.features.persistence.save_feature_registry import (
    save_feature_registry,
)
from ml_service.backend.configs.features.utils.paths import get_registry_path
from ml_service.backend.configs.features.utils.registry import registry_entry_exists
from ml_service.backend.configs.features.validation.validate_feature_config import (
    validate_feature_config,
)
from ml_service.backend.configs.loading.load_yaml_and_add_lineage import load_yaml_and_add_lineage
from ml_service.backend.main import limiter

router = APIRouter(prefix="/features", tags=["features"])

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root

@router.post("/validate", status_code=200)
@limiter.limit("1/15seconds")
def validate_yaml(payload: dict, request: Request):
    """Validate feature config and check registry collisions."""

    try:
        name = payload.get("name")
        version = payload.get("version")
        yaml_text = payload.get("config")

        if not name or not version:
            raise ValueError("Missing feature set name or version.")

        if not yaml_text:
            raise ValueError("Missing feature_config payload")

        data_with_lineage = load_yaml_and_add_lineage(yaml_text)

        validated = validate_feature_config(data_with_lineage)

        registry_path = get_registry_path(Path(repo_root))

        exists = registry_entry_exists(name, version, registry_path)

        return {
            "valid": True,
            "exists": exists,
            "normalized": validated.model_dump(mode="json"),
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}


@router.post("/write", status_code=201)
@limiter.limit("1/minute")
def write_yaml(payload: dict, request: Request):
    """Validate and write feature config to registry."""

    try:
        name = payload.get("name")
        version = payload.get("version")
        yaml_text = payload.get("config")

        if not name or not version:
            raise ValueError("Missing feature set name or version.")

        if not yaml_text:
            raise ValueError("Missing feature_config payload")

        data_with_lineage = load_yaml_and_add_lineage(yaml_text)

        validated = validate_feature_config(data_with_lineage)

        registry_path = get_registry_path(Path(repo_root))

        if registry_entry_exists(name, version, registry_path):
            return {
                "status": "exists",
                "message": f"{name}/{version} already exists in registry",
            }

        result = save_feature_registry(
            name,
            version,
            validated_config=validated,
            registry_path=registry_path
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
