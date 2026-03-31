"""Tests for the promotion_thresholds router endpoints (validate + write)."""

from __future__ import annotations

import pytest
from fastapi import Request


def _fake_validated():
    class FakeValidated:
        def model_dump(self, mode="json"):
            return {"ok": True}

    return FakeValidated()


def _req():
    return Request({"type": "http", "method": "POST", "path": "/", "headers": []})


def test_validate_yaml_success_exists(monkeypatch):
    import ml_service.backend.routers.promotion_thresholds as pt_mod

    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.load_yaml_and_add_lineage",
        lambda text: {"foo": "bar"},
    )

    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.validate_config_payload",
        lambda d: _fake_validated(),
    )

    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.check_thresholds_exist",
        lambda path, pt, seg: (True, {"t": 1}),
    )

    orig = getattr(pt_mod.validate_yaml, "__wrapped__", pt_mod.validate_yaml)
    payload = {"config": "cfg", "problem_type": "p", "segment": "s"}
    res = orig(payload, _req())
    assert res["valid"] is True
    assert res["exists"] is True
    assert res["normalized"]["ok"] is True


def test_validate_yaml_missing_config_returns_error():
    import ml_service.backend.routers.promotion_thresholds as pt_mod

    orig = getattr(pt_mod.validate_yaml, "__wrapped__", pt_mod.validate_yaml)
    res = orig({}, _req())
    assert res["valid"] is False
    assert "Missing config payload" in res["error"]


def test_validate_yaml_missing_fields_returns_error(monkeypatch):
    import ml_service.backend.routers.promotion_thresholds as pt_mod

    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.load_yaml_and_add_lineage",
        lambda text: {"foo": "bar"},
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.validate_config_payload",
        lambda d: _fake_validated(),
    )

    orig = getattr(pt_mod.validate_yaml, "__wrapped__", pt_mod.validate_yaml)
    res = orig({"config": "x"}, _req())
    assert res["valid"] is False
    assert "Missing required fields" in res["error"]


def test_write_yaml_exists(monkeypatch):
    import ml_service.backend.routers.promotion_thresholds as pt_mod

    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.load_yaml_and_add_lineage",
        lambda text: {"foo": "bar"},
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.validate_config_payload",
        lambda d: _fake_validated(),
    )

    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.check_thresholds_exist",
        lambda path, pt, seg: (True, {"t": 1}),
    )

    orig = getattr(pt_mod.write_yaml, "__wrapped__", pt_mod.write_yaml)
    payload = {"config": "cfg", "problem_type": "p", "segment": "s"}
    res = orig(payload, _req())
    assert res["status"] == "exists"


def test_write_yaml_written_and_save_called(monkeypatch, tmp_path):
    import ml_service.backend.routers.promotion_thresholds as pt_mod

    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.load_yaml_and_add_lineage",
        lambda text: {"foo": "bar"},
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.validate_config_payload",
        lambda d: _fake_validated(),
    )

    pt_mod.repo_root = str(tmp_path)

    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.check_thresholds_exist",
        lambda path, pt, seg: (False, {"min": 0, "max": 1}),
    )

    called = {}

    def _save(thresholds, validated, config_path, problem_type, segment):
        called["args"] = (thresholds, validated, str(config_path), problem_type, segment)

    monkeypatch.setattr("ml_service.backend.routers.promotion_thresholds.save_promotion_thresholds", _save)

    orig = getattr(pt_mod.write_yaml, "__wrapped__", pt_mod.write_yaml)
    payload = {"config": "cfg", "problem_type": "p", "segment": "s"}
    res = orig(payload, _req())
    assert res["success"] == "written"
    assert "path" in res
    assert "args" in called


def test_write_yaml_save_failure_raises(monkeypatch, tmp_path):
    import ml_service.backend.routers.promotion_thresholds as pt_mod
    from fastapi import HTTPException

    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.load_yaml_and_add_lineage",
        lambda text: {"foo": "bar"},
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.validate_config_payload",
        lambda d: _fake_validated(),
    )

    pt_mod.repo_root = str(tmp_path)

    monkeypatch.setattr(
        "ml_service.backend.routers.promotion_thresholds.check_thresholds_exist",
        lambda path, pt, seg: (False, {"min": 0}),
    )

    def _bad_save(*args, **kwargs):
        raise RuntimeError("no space")

    monkeypatch.setattr("ml_service.backend.routers.promotion_thresholds.save_promotion_thresholds", _bad_save)

    orig = getattr(pt_mod.write_yaml, "__wrapped__", pt_mod.write_yaml)
    payload = {"config": "cfg", "problem_type": "p", "segment": "s"}
    with pytest.raises(HTTPException):
        orig(payload, _req())
