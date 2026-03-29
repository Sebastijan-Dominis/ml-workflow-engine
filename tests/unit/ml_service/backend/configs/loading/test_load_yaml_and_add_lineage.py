import importlib

import pytest


def test_load_yaml_and_add_lineage_injects_timestamp(monkeypatch):
    mod = importlib.import_module(
        "ml_service.backend.configs.loading.load_yaml_and_add_lineage"
    )
    # patch the underlying timestamp generator for deterministic output
    ts_mod = importlib.import_module(
        "ml_service.backend.configs.formatting.timestamp"
    )
    monkeypatch.setattr(ts_mod, "utc_timestamp", lambda: "2026-03-29T12:00:00Z")

    yaml_text = "lineage: {}\nfoo: bar\n"
    res = mod.load_yaml_and_add_lineage(yaml_text)
    assert "lineage" in res
    assert res["lineage"]["created_at"] == "2026-03-29T12:00:00Z"
    assert res["foo"] == "bar"


def test_load_yaml_and_add_lineage_missing_lineage_raises():
    mod = importlib.import_module(
        "ml_service.backend.configs.loading.load_yaml_and_add_lineage"
    )
    with pytest.raises(ValueError):
        mod.load_yaml_and_add_lineage("foo: bar\n")
