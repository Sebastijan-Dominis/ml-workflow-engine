"""Unit tests for `ml_service.backend.configs.modeling.utils.paths`.

These tests exercise `compute_paths` and `check_paths` logic with
lightweight dummy objects and monkeypatched filesystem checks.
"""
from __future__ import annotations

from typing import Any

import pytest
from ml_service.backend.configs.modeling.utils import paths as paths_mod


def _make_dummy_validated(problem: str = "prob", segment_name: str = "seg", version: str = "v1") -> Any:
    class DummySeg:
        def __init__(self, name: str) -> None:
            self.name = name

    class DummyModelSpecs:
        def __init__(self, problem: str, segment: Any, version: str) -> None:
            self.problem = problem
            self.segment = segment
            self.version = version

    class V:
        model_specs: Any

    v = V()
    v.model_specs = DummyModelSpecs(problem, DummySeg(segment_name), version)
    return v


def test_compute_paths_returns_expected_paths() -> None:
    v = _make_dummy_validated("myprob", "myseg", "vv")
    result = paths_mod.compute_paths(v)

    expected_model = f"{paths_mod.repo_root}/configs/model_specs/myprob/myseg/vv.yaml"
    expected_search = f"{paths_mod.repo_root}/configs/search/myprob/myseg/vv.yaml"
    expected_train = f"{paths_mod.repo_root}/configs/train/myprob/myseg/vv.yaml"

    assert result.model_specs == expected_model
    assert result.search == expected_search
    assert result.training == expected_train


def test_check_paths_raises_on_existing_model_spec(monkeypatch) -> None:
    v = _make_dummy_validated("prob", "seg", "v1")
    paths = paths_mod.compute_paths(v)

    # Simulate that only model_specs path already exists
    def fake_exists(p: str) -> bool:
        return p == paths.model_specs

    monkeypatch.setattr(paths_mod.os.path, "exists", fake_exists)

    with pytest.raises(FileExistsError) as exc:
        paths_mod.check_paths(v)

    assert paths.model_specs in str(exc.value)


def test_check_paths_returns_paths_when_none_exist(monkeypatch) -> None:
    v = _make_dummy_validated("prob", "seg", "v1")

    monkeypatch.setattr(paths_mod.os.path, "exists", lambda p: False)

    res = paths_mod.check_paths(v)
    # ensure returned object matches compute_paths output
    expected = paths_mod.compute_paths(v)
    assert res.model_specs == expected.model_specs
    assert res.search == expected.search
    assert res.training == expected.training


def test_check_paths_raises_on_existing_search(monkeypatch) -> None:
    v = _make_dummy_validated("prob", "seg", "v1")
    paths = paths_mod.compute_paths(v)

    # Simulate that only search path already exists
    def fake_exists(p: str) -> bool:
        return p == paths.search

    monkeypatch.setattr(paths_mod.os.path, "exists", fake_exists)

    with pytest.raises(FileExistsError) as exc:
        paths_mod.check_paths(v)

    assert paths.search in str(exc.value)


def test_check_paths_raises_on_existing_training(monkeypatch) -> None:
    v = _make_dummy_validated("prob", "seg", "v1")
    paths = paths_mod.compute_paths(v)

    # Simulate that only training path already exists
    def fake_exists(p: str) -> bool:
        return p == paths.training

    monkeypatch.setattr(paths_mod.os.path, "exists", fake_exists)

    with pytest.raises(FileExistsError) as exc:
        paths_mod.check_paths(v)

    assert paths.training in str(exc.value)
