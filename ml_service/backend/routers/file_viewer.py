"""A FastAPI router for viewing file contents on the backend, used by the frontend for visualization."""
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from ml.utils.loaders import load_json, load_yaml

from ml_service.backend.main import limiter

router: APIRouter = APIRouter(prefix="/file_viewer", tags=["file_viewer"])

@router.post("/load", status_code=200)
@limiter.limit("1/5seconds")
def load_file(payload: dict, request: Request):
    """
    Load a YAML or JSON file given its path.
    payload: {"path": "<absolute or relative path>"}
    """
    path_str = payload.get("path")
    if not path_str:
        raise HTTPException(status_code=400, detail="Missing 'path' in request payload")

    normalized_path_str = path_str.replace("\\", "/")
    path = Path(normalized_path_str)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        if path.suffix in [".yaml", ".yml"]:
            data = load_yaml(path)
            text = yaml.safe_dump(data, sort_keys=False)
            mode = "yaml"
        elif path.suffix == ".json":
            data = load_json(path)
            import json
            text = json.dumps(data, indent=2)
            mode = "json"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type, only .yaml/.yml/.json")

        return {"content": text, "mode": mode, "path": str(path)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
