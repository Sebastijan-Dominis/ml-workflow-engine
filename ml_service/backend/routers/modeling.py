"""A module for handling modeling-related API endpoints, including validation and writing of YAML configurations."""
from fastapi import APIRouter, HTTPException, Request

from ml_service.backend.configs.modeling.loading.load_all_yamls_and_add_lineage import (
    load_all_yamls_and_add_lineage,
)
from ml_service.backend.configs.modeling.persistence.save_all_configs import save_all_configs
from ml_service.backend.configs.modeling.utils.paths import check_paths
from ml_service.backend.configs.modeling.validation.validate_all_configs import validate_all_configs
from ml_service.backend.main import limiter

router = APIRouter(prefix="/modeling", tags=["modeling"])

@router.post("/validate", status_code=200)
@limiter.limit("1/15seconds")
def validate_yaml(payload: dict, request: Request):
    """Validate the provided YAML configurations for modeling."""
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
    """Validate and write the provided YAML configurations for modeling."""
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
