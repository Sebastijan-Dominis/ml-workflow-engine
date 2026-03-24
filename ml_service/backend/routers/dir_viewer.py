"""A FastAPI router for viewing directory structure on the backend, used by the frontend for visualization."""
import os
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request

from ml_service.backend.dir_viewer.utils.build_tree import build_tree
from ml_service.backend.main import limiter

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

router: APIRouter = APIRouter(prefix="/dir_viewer", tags=["dir_viewer"])

@router.post("/load", status_code=200)
@limiter.limit("10/minute")
def load_dir(payload: dict, request: Request):
    """
    payload: {"path": "<relative path from repo root>"}
    Returns folder tree as YAML string.
    """
    path_str = payload.get("path")
    if not path_str:
        raise HTTPException(status_code=400, detail="Missing 'path' in payload")

    # Make the path relative to repo root
    path = (Path(repo_root) / path_str).resolve()

    # Safety: prevent access outside repo
    try:
        path.relative_to(repo_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Cannot access directories outside repo root") from None

    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    tree = build_tree(path)

    # Convert tree to YAML for frontend display
    tree_yaml = yaml.safe_dump(tree, sort_keys=False)

    return {"tree": tree, "tree_yaml": tree_yaml, "path": str(path)}
