import os

from fastapi import APIRouter, HTTPException, Request

from ml_service.backend.configs.loading.load_yaml_and_add_lineage import load_yaml_and_add_lineage
from ml_service.backend.configs.persistence.save_config import save_config
from ml_service.backend.configs.pipeline_cfg.utils.get_config_path import get_config_path
from ml_service.backend.configs.pipeline_cfg.validation.validate_config_payload import (
    validate_config_payload,
)
from ml_service.backend.main import limiter

router: APIRouter = APIRouter(prefix="/pipeline_cfg", tags=["pipeline_cfg"])

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root

@router.post("/validate", status_code=200)
@limiter.limit("1/15seconds")
def validate_yaml(payload: dict, request: Request):
    """Validate pipeline config."""

    try:
        yaml_text = payload.get("config")

        if not yaml_text:
            raise ValueError("Missing config payload.")

        data_dict = load_yaml_and_add_lineage(yaml_text)
        validated = validate_config_payload(data_dict)

        data_type = payload.get("data_type")
        algorithm = payload.get("algorithm")
        pipeline_version = data_dict.get("version")

        if not all([data_type, algorithm, pipeline_version]):
            raise ValueError("Missing required fields: data_type, algorithm, or version.")

        config_path = get_config_path(
            repo_root=repo_root,
            data_type=str(data_type),
            algorithm=str(algorithm),
            pipeline_version=str(pipeline_version)
        )
        exists = config_path.exists()

        return {
            "valid": True,
            "exists": exists,
            "normalized": validated.model_dump(mode="json")
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}

@router.post("/write", status_code=201)
@limiter.limit("1/minute; 30/day")
def write_yaml(payload: dict, request: Request):
    """Validate and write pipeline config atomically."""

    try:
        yaml_text = payload.get("config")

        if not yaml_text:
            raise ValueError("Missing config payload.")

        data_dict = load_yaml_and_add_lineage(yaml_text)
        _ = validate_config_payload(data_dict)

        data_type = payload.get("data_type")
        algorithm = payload.get("algorithm")
        pipeline_version = data_dict.get("version")

        if not all([data_type, algorithm, pipeline_version]):
            raise ValueError("Missing required fields: data_type, algorithm, or version.")

        config_path = get_config_path(
            repo_root=repo_root,
            data_type=str(data_type),
            algorithm=str(algorithm),
            pipeline_version=str(pipeline_version)
        )

        if config_path.exists():
            return {
                "status": "exists",
                "message": f"{data_type}/{algorithm}/{pipeline_version} already exists.",
            }

        save_config(config=data_dict, config_path=config_path)

        return {"success": "written", "path": str(config_path)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
