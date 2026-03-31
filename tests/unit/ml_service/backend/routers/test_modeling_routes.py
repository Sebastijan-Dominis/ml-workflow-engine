"""Tests for the `ml_service.backend.routers.modeling` FastAPI router.

These tests monkeypatch the internal helpers to keep the tests lightweight
and focus on routing/response behavior.
"""
from __future__ import annotations

from dataclasses import dataclass

import ml_service.backend.routers.modeling as modeling_router
import pytest
from fastapi import Request


class _DummyModel:
    def __init__(self, name: str) -> None:
        self._name = name

    def model_dump(self, mode: str = "json", exclude: dict | None = None) -> dict:
        return {"name": self._name, "mode": mode, "exclude": bool(exclude)}


class _DummyValidated:
    def __init__(self) -> None:
        self.model_specs = _DummyModel("model_specs")
        self.search = _DummyModel("search")
        self.training = _DummyModel("training")


@dataclass
class _DummyPaths:
    model_specs: str
    search: str
    training: str


def test_validate_yaml_success(fastapi_client, monkeypatch) -> None:
    # Arrange: patch loaders/validators/paths to return predictable objects
    monkeypatch.setattr(modeling_router, "load_all_yamls_and_add_lineage", lambda payload: {"ok": True})
    monkeypatch.setattr(modeling_router, "validate_all_configs", lambda data: _DummyValidated())
    monkeypatch.setattr(modeling_router, "check_paths", lambda validated: None)

    # Act - call the undecorated function to bypass slowapi rate limiting in tests
    orig = getattr(modeling_router.validate_yaml, "__wrapped__", modeling_router.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig({"model_specs": "x", "search": "y", "training": "z"}, req)

    # Assert
    assert j["valid"] is True
    assert "normalized" in j and "model_specs" in j["normalized"]


def test_validate_yaml_error_returns_valid_false(fastapi_client, monkeypatch) -> None:
    # Simulate loader raising an error
    def _bad(_):
        raise ValueError("bad yaml")

    monkeypatch.setattr(modeling_router, "load_all_yamls_and_add_lineage", _bad)

    orig = getattr(modeling_router.validate_yaml, "__wrapped__", modeling_router.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig({"model_specs": "x", "search": "y", "training": "z"}, req)
    assert j["valid"] is False
    assert "bad yaml" in j["error"]


def test_write_yaml_success(fastapi_client, monkeypatch) -> None:
    monkeypatch.setattr(modeling_router, "load_all_yamls_and_add_lineage", lambda payload: {"ok": True})
    monkeypatch.setattr(modeling_router, "validate_all_configs", lambda data: _DummyValidated())

    paths = _DummyPaths(model_specs="p1.yaml", search="p2.yaml", training="p3.yaml")
    monkeypatch.setattr(modeling_router, "check_paths", lambda validated: paths)

    saved = {}

    def _save(validated, pths):
        # record that save_all_configs was called with the validated object and returned paths
        saved["called"] = True
        saved["paths"] = pths

    monkeypatch.setattr(modeling_router, "save_all_configs", _save)

    orig = getattr(modeling_router.write_yaml, "__wrapped__", modeling_router.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig({"model_specs": "x", "search": "y", "training": "z"}, req)
    assert j["paths"]["model_specs"] == "p1.yaml"
    assert saved.get("called") is True


def test_write_yaml_failure_raises_http_exception(fastapi_client, monkeypatch) -> None:
    monkeypatch.setattr(modeling_router, "load_all_yamls_and_add_lineage", lambda payload: {"ok": True})
    monkeypatch.setattr(modeling_router, "validate_all_configs", lambda data: _DummyValidated())

    monkeypatch.setattr(modeling_router, "check_paths", lambda validated: _DummyPaths("a", "b", "c"))

    def _bad_save(validated, paths):
        raise RuntimeError("disk full")

    monkeypatch.setattr(modeling_router, "save_all_configs", _bad_save)

    orig = getattr(modeling_router.write_yaml, "__wrapped__", modeling_router.write_yaml)
    with pytest.raises(Exception) as exc:
        req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
        orig({"model_specs": "x", "search": "y", "training": "z"}, req)

    assert "disk full" in str(exc.value)
