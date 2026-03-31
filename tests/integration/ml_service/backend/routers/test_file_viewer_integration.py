"""Integration tests for the `file_viewer` FastAPI router.

These tests exercise the endpoint that loads YAML/JSON files by path and
returns their content and detected mode.
"""

import json
from pathlib import Path
from typing import Any

import yaml


def test_file_viewer_load_yaml(tmp_path: Path, fastapi_client: Any) -> None:
    # Create a temporary YAML file
    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "sample.yaml"
    cfg = {"a": 1, "b": [1, 2, 3]}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    resp = fastapi_client.post("/file_viewer/load", json={"path": str(cfg_path)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "yaml"
    # parse returned content to verify payload round-trips
    parsed = yaml.safe_load(body["content"])
    assert parsed == cfg


def test_file_viewer_load_json(tmp_path: Path, fastapi_client: Any) -> None:
    data_dir = tmp_path / "datajson"
    data_dir.mkdir()
    json_path = data_dir / "sample.json"
    payload = {"x": 10, "y": ["a"]}
    json_path.write_text(json.dumps(payload))

    resp = fastapi_client.post("/file_viewer/load", json={"path": str(json_path)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "json"
    parsed = json.loads(body["content"])
    assert parsed == payload
