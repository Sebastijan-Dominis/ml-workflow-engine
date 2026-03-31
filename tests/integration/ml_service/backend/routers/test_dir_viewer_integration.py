"""Integration tests for the `dir_viewer` FastAPI router.

These tests create a small directory under the repository and request a
directory tree for it via the router, validating the response structure.
"""

import contextlib
import shutil
import uuid
from pathlib import Path
from typing import Any


def test_dir_viewer_load(tmp_path: Path, fastapi_client: Any) -> None:
    # Create a unique folder inside repo under tests/ so the router can access it
    repo_root = Path.cwd()
    unique_name = f"tmp_dir_{uuid.uuid4().hex[:8]}"
    target = repo_root / "tests" / unique_name
    try:
        target.mkdir(parents=True, exist_ok=False)
        (target / "a.txt").write_text("hello")
        sub = target / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("x")

        resp = fastapi_client.post("/dir_viewer/load", json={"path": f"tests/{unique_name}"})
        assert resp.status_code == 200
        body = resp.json()
        assert "tree" in body and "tree_yaml" in body
        tree = body["tree"]
        # Expect top-level file and subdirectory
        assert "a.txt" in tree
        assert "sub" in tree
    finally:
        # Best-effort cleanup
        with contextlib.suppress(Exception):
            shutil.rmtree(target)
