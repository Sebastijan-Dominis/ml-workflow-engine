"""Tests for `ml_service.backend.configs.persistence.save_config`."""

from __future__ import annotations

import importlib

import pytest
import yaml
from fastapi import HTTPException
from ml_service.backend.configs.persistence.save_config import save_config


def test_save_config_success(tmp_path) -> None:
    cfg = {"a": 1}
    p = tmp_path / "cfg.yaml"
    save_config(cfg, p)
    assert p.exists()
    assert yaml.safe_load(p.read_text(encoding="utf-8")) == cfg


def test_save_config_failure_cleans_tmp(tmp_path, monkeypatch) -> None:
    cfg = {"b": 2}
    p = tmp_path / "cfg.yaml"
    tmp_file = p.parent / f"{p.name}.tmp"

    def _bad_replace(src, dst):
        raise OSError("boom")

    monkeypatch.setattr(
        "ml_service.backend.configs.persistence.save_config.os.replace",
        _bad_replace,
    )

    with pytest.raises(HTTPException) as exc:
        save_config(cfg, p)

    # tmp file should be removed by the exception handler
    assert not tmp_file.exists()
    assert exc.value.status_code == 500


def test_save_config_open_fails_no_tmp(tmp_path, monkeypatch) -> None:
    cfg = {"c": 3}
    p = tmp_path / "cfg2.yaml"
    tmp_file = p.parent / f"{p.name}.tmp"

    def _bad_open(*args, **kwargs):
        raise OSError("open failed")
    monkeypatch.setattr("builtins.open", _bad_open)

    with pytest.raises(HTTPException) as exc:
        save_config(cfg, p)

    assert not tmp_file.exists()
    assert exc.value.status_code == 500
def test_save_config_writes_file(tmp_path) -> None:
    sc = importlib.import_module("ml_service.backend.configs.persistence.save_config")
    cfg = {"alpha": 1, "nested": {"x": "y"}}
    cp = tmp_path / "cfgs" / "cfg.yaml"
    sc.save_config(cfg, cp)
    assert cp.exists()
    loaded = yaml.safe_load(cp.read_text(encoding="utf-8"))
    assert loaded == cfg
    # ensure no tmp file left behind
    assert not (cp.parent / f"{cp.name}.tmp").exists()


def test_save_config_failure_cleans_tmp_via_module(tmp_path, monkeypatch) -> None:
    sc = importlib.import_module("ml_service.backend.configs.persistence.save_config")
    cp = tmp_path / "cfgs" / "cfg.yaml"
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cp.parent / f"{cp.name}.tmp"
    tmp_file.write_text("will be removed")

    def raise_replace(a, b):
        raise OSError("boom")

    monkeypatch.setattr(sc.os, "replace", raise_replace)

    with pytest.raises(sc.HTTPException) as excinfo:
        sc.save_config({"a": 1}, cp)

    assert not tmp_file.exists()
    assert excinfo.value.status_code == 500
