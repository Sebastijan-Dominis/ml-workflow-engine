import importlib
import re

import pytest


def test_add_timestamp_raises_on_missing_lineage_key():
    mod = importlib.import_module("ml_service.backend.configs.formatting.timestamp")
    data = {}
    try:
        mod.add_timestamp(data, "lineage")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_add_timestamp_sets_iso_created_at():
    mod = importlib.import_module("ml_service.backend.configs.formatting.timestamp")
    data = {"lineage": {"created_by": "tester"}}
    out = mod.add_timestamp(data, "lineage")
    assert "created_at" in out["lineage"]
    ts = out["lineage"]["created_at"]
    assert isinstance(ts, str)
    # basic ISO-like check: contains T and ends with Z
    assert "T" in ts and ts.endswith("Z")


def test_utc_timestamp_format():
    mod = importlib.import_module("ml_service.backend.configs.formatting.timestamp")
    ts = mod.utc_timestamp()
    assert isinstance(ts, str)
    # basic pattern YYYY-MM-DDTHH:MM:SSZ
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts)


def test_add_timestamp_sets_created_at(monkeypatch):
    mod = importlib.import_module(
        "ml_service.backend.configs.formatting.timestamp"
    )
    monkeypatch.setattr(mod, "utc_timestamp", lambda: "2026-03-29T12:00:00Z")
    data = {"lineage": {}}
    res = mod.add_timestamp(data, "lineage")
    assert "created_at" in data["lineage"]
    assert data["lineage"]["created_at"] == "2026-03-29T12:00:00Z"
    assert res is data


def test_add_timestamp_missing_key_raises():
    mod = importlib.import_module(
        "ml_service.backend.configs.formatting.timestamp"
    )
    with pytest.raises(ValueError):
        mod.add_timestamp({}, "lineage")
