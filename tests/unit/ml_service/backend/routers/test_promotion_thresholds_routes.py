"""Tests for `ml_service.backend.routers.promotion_thresholds` write branches."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


def test_write_promotion_missing_config_payload_raises() -> None:
    mod = importlib.import_module("ml_service.backend.routers.promotion_thresholds")
    orig = getattr(mod.write_yaml, "__wrapped__", mod.write_yaml)
    from fastapi import Request

    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(Exception) as exc:
        orig({}, req)
    assert "Missing config payload" in str(exc.value)


def test_write_promotion_missing_required_fields(monkeypatch) -> None:
    mod = importlib.import_module("ml_service.backend.routers.promotion_thresholds")
    monkeypatch.setattr(mod, "load_yaml_and_add_lineage", lambda txt: {"v": 1})
    monkeypatch.setattr(mod, "validate_config_payload", lambda d: SimpleNamespace())

    orig = getattr(mod.write_yaml, "__wrapped__", mod.write_yaml)
    from fastapi import Request

    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(Exception) as exc:
        orig({"config": "cfg"}, req)
    assert "Missing required fields" in str(exc.value)
