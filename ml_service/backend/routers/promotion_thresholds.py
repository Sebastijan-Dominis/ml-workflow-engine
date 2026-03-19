import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ml_service.backend.configs.loading.load_yaml_and_add_lineage import load_yaml_and_add_lineage
from ml_service.backend.configs.promotion_thresholds.persistence.save_promotion_thresholds import (
    save_promotion_thresholds,
)
from ml_service.backend.configs.promotion_thresholds.utils.check_thresholds_exist import (
    check_thresholds_exist,
)
from ml_service.backend.configs.promotion_thresholds.validation.validate_config_payload import (
    validate_config_payload,
)
from ml_service.backend.main import limiter

router = APIRouter(prefix="/promotion_thresholds", tags=["promotion_thresholds"])


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

        problem_type = payload.get("problem_type")
        segment = payload.get("segment")

        if not all([problem_type, segment]):
            raise ValueError("Missing required fields: problem_type or segment.")

        config_path = Path(repo_root) / "configs" / "promotion" /"thresholds.yaml"

        exists, _ = check_thresholds_exist(config_path, str(problem_type), str(segment))

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

        problem_type = payload.get("problem_type")
        segment = payload.get("segment")

        if not all([problem_type, segment]):
            raise ValueError("Missing required fields: problem_type or segment.")

        config_path = Path(repo_root) / "configs" / "promotion" / "thresholds.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        exist, thresholds = check_thresholds_exist(config_path, str(problem_type), str(segment))

        if exist:
            return {
                "status": "exists",
                "message": f"Thresholds for {problem_type}/{segment} already exist.",
            }

        save_promotion_thresholds(
            thresholds=thresholds,
            validated=validated,
            config_path=config_path,
            problem_type=str(problem_type),
            segment=str(segment)
        )

        return {"success": "written", "path": str(config_path)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
