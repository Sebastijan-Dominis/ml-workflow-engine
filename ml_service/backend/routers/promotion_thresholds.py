import copy
import os
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from ml.promotion.config.models import PromotionThresholds

from ml_service.backend.configs.formatting.timestamp import add_timestamp
from ml_service.backend.main import limiter

router = APIRouter(prefix="/promotion_thresholds", tags=["promotion_thresholds"])


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root

def validate_config_payload(payload: dict) -> PromotionThresholds:
    """Validate the incoming payload against the PromotionThresholds schema."""
    return PromotionThresholds(**payload)

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

def check_thresholds_exist(
    config_path: Path,
    problem_type: str,
    segment: str
) -> tuple[bool, dict]:
    """Check if thresholds already exist for the given problem type and segment."""
    if not config_path.exists():
        return False, {}

    with open(config_path) as f:
        thresholds = yaml.safe_load(f) or {}

    target_thresholds = thresholds.get(problem_type, {}).get(segment)

    return target_thresholds is not None, thresholds

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

        thresholds_new = copy.deepcopy(thresholds) if thresholds else {}

        thresholds_new.setdefault(problem_type, {})[segment] = validated.model_dump(mode="json")

        tmp_path = config_path.parent / f"{config_path.name}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(thresholds_new, f, sort_keys=False)
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
