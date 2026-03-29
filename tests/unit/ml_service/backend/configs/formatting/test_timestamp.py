import importlib

import pytest


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
