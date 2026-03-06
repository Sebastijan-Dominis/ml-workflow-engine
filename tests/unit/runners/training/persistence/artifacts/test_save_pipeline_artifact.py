"""Unit tests for training pipeline artifact persistence helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from ml.exceptions import PersistenceError
from ml.runners.training.persistence.artifacts import save_pipeline as module
from sklearn.pipeline import Pipeline

pytestmark = pytest.mark.unit


def test_save_pipeline_writes_pipeline_joblib_and_returns_pipeline_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist pipeline to ``pipeline.joblib`` and return the persisted artifact path."""
    captured: dict[str, object] = {}

    def _dump(pipeline: object, pipeline_file: Path) -> None:
        captured["pipeline"] = pipeline
        captured["path"] = pipeline_file

    monkeypatch.setattr(module.joblib, "dump", _dump)

    pipeline_obj = cast(Pipeline, SimpleNamespace(name="stub-pipeline"))
    saved_path = module.save_pipeline(pipeline_obj, tmp_path)

    assert saved_path == tmp_path / "pipeline.joblib"
    assert captured["pipeline"] is pipeline_obj
    assert captured["path"] == tmp_path / "pipeline.joblib"


def test_save_pipeline_wraps_joblib_failures_as_persistence_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Wrap pipeline serialization failures as ``PersistenceError`` with path context."""

    def _raise_dump_error(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(module.joblib, "dump", _raise_dump_error)

    with pytest.raises(PersistenceError, match="Failed to save pipeline"):
        module.save_pipeline(cast(Pipeline, SimpleNamespace()), tmp_path)
