import os
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from ml.pipelines.models import PipelineConfig

from ml_service.backend.configs.formatting.timestamp import add_timestamp
from ml_service.backend.main import limiter

router = APIRouter(prefix="/pipeline_cfg", tags=["pipeline_cfg"])


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root

def get_config_path(data_type: str, algorithm: str, pipeline_version: str) -> Path:
    """Construct the file path for a given config type, algorithm, and version."""
    return Path(repo_root) / "configs" / "pipelines" / data_type / algorithm / f"{pipeline_version}.yaml"

def validate_config_payload(payload: dict) -> PipelineConfig:
    """Validate the incoming payload against the PipelineConfig schema."""
    return PipelineConfig(**payload)

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
        validated = validate_config_payload(data_dict)

        data_type = payload.get("data_type")
        algorithm = payload.get("algorithm")
        pipeline_version = data_dict.get("version")

        if not all([data_type, algorithm, pipeline_version]):
            raise ValueError("Missing required fields: data_type, algorithm, or version.")

        config_path = get_config_path(
            data_type=str(data_type),
            algorithm=str(algorithm),
            pipeline_version=str(pipeline_version)
        )

        if config_path.exists():
            return {
                "status": "exists",
                "message": f"{data_type}/{algorithm}/{pipeline_version} already exists.",
            }

        config_path.parent.mkdir(parents=True, exist_ok=True)


        tmp_path = config_path.parent / f"{config_path.name}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(validated.model_dump(mode="json"), f, sort_keys=False)
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

        return {"success": "written", "path": str(config_path)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
