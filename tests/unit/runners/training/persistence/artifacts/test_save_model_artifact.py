"""Unit tests for training model artifact persistence helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from ml.exceptions import PersistenceError
from ml.runners.training.persistence.artifacts import save_model as module

pytestmark = pytest.mark.unit


def test_save_model_writes_model_joblib_and_returns_model_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist model to ``model.joblib`` and return the persisted artifact path."""
    captured: dict[str, object] = {}

    def _dump(model: object, model_file: Path) -> None:
        captured["model"] = model
        captured["path"] = model_file

    monkeypatch.setattr(module.joblib, "dump", _dump)

    model_obj = SimpleNamespace(name="stub-model")
    saved_path = module.save_model(model_obj, tmp_path)

    assert saved_path == tmp_path / "model.joblib"
    assert captured["model"] is model_obj
    assert captured["path"] == tmp_path / "model.joblib"


def test_save_model_wraps_joblib_failures_as_persistence_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Wrap joblib serialization failures as ``PersistenceError`` with file context."""

    def _raise_dump_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(module.joblib, "dump", _raise_dump_error)

    with pytest.raises(PersistenceError, match="Failed to save model"):
        module.save_model(SimpleNamespace(), tmp_path)
