import importlib

import pytest


def test_load_registry_missing_raises(tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.utils.registry"
    )
    p = tmp_path / "nope.yaml"
    assert not p.exists()
    with pytest.raises(RuntimeError, match="Feature registry missing"):
        mod.load_registry(p)


def test_load_registry_empty_raises(tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.utils.registry"
    )
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(RuntimeError, match="empty or corrupted"):
        mod.load_registry(p)


def test_load_registry_non_dict_raises(tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.utils.registry"
    )
    p = tmp_path / "list.yaml"
    p.write_text("- one\n- two\n")
    with pytest.raises(RuntimeError, match="must be a dict"):
        mod.load_registry(p)


def test_load_registry_returns_dict(tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.utils.registry"
    )
    p = tmp_path / "reg.yaml"
    p.write_text("foo:\n  bar: 1\n")
    res = mod.load_registry(p)
    assert isinstance(res, dict)
    assert res == {"foo": {"bar": 1}}


def test_registry_entry_exists_true(tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.utils.registry"
    )
    p = tmp_path / "reg.yaml"
    p.write_text("featA:\n  v1: {}\n")
    assert mod.registry_entry_exists("featA", "v1", p) is True


def test_registry_entry_exists_false(tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.utils.registry"
    )
    p = tmp_path / "reg.yaml"
    p.write_text("featA:\n  v1: {}\n")
    assert mod.registry_entry_exists("featA", "v2", p) is False


def test_registry_entry_exists_name_missing(tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.utils.registry"
    )
    p = tmp_path / "reg.yaml"
    p.write_text("featB:\n  v1: {}\n")
    assert mod.registry_entry_exists("unknown", "v1", p) is False
