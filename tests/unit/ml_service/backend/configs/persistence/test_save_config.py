import importlib

import pytest
import yaml


def test_save_config_writes_file(tmp_path):
    sc = importlib.import_module(
        "ml_service.backend.configs.persistence.save_config"
    )
    cfg = {"alpha": 1, "nested": {"x": "y"}}
    cp = tmp_path / "cfgs" / "cfg.yaml"
    sc.save_config(cfg, cp)
    assert cp.exists()
    loaded = yaml.safe_load(cp.read_text(encoding="utf-8"))
    assert loaded == cfg
    assert not (cp.parent / f"{cp.name}.tmp").exists()


def test_save_config_failure_cleans_tmp(tmp_path, monkeypatch):
    sc = importlib.import_module(
        "ml_service.backend.configs.persistence.save_config"
    )
    cp = tmp_path / "cfgs" / "cfg.yaml"

    def raise_replace(a, b):
        raise OSError("boom")

    monkeypatch.setattr(sc.os, "replace", raise_replace)

    with pytest.raises(sc.HTTPException) as excinfo:
        sc.save_config({"a": 1}, cp)

    tmp_file = cp.parent / f"{cp.name}.tmp"
    assert not tmp_file.exists()
    assert excinfo.value.status_code == 500
