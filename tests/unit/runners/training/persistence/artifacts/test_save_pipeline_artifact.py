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
    captured_pipeline: object | None = None
    captured_path: Path | None = None

    def _dump(pipeline: object, pipeline_file: Path) -> None:
        nonlocal captured_pipeline, captured_path
        captured_pipeline = pipeline
        captured_path = pipeline_file

    monkeypatch.setattr(module.joblib, "dump", _dump)

    pipeline_obj = cast(Pipeline, SimpleNamespace(name="stub-pipeline"))
    saved_path = module.save_pipeline(pipeline_obj, tmp_path)

    assert saved_path == tmp_path / "pipeline.joblib"
    assert captured_pipeline is pipeline_obj
    assert captured_path is not None
    dumped_path = captured_path
    assert dumped_path.parent == tmp_path
    assert dumped_path.name.startswith("pipeline.")
    assert dumped_path.suffixes[-2:] == [".joblib", ".tmp"]
    assert saved_path.exists()


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


def test_save_pipeline_creates_missing_parent_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Create artifact directory before dumping pipeline when target path is missing."""
    target_dir = tmp_path / "nested" / "artifacts"

    def _dump_to_path(_pipeline: object, pipeline_file: Path) -> None:
        pipeline_file.write_text("pipeline", encoding="utf-8")

    monkeypatch.setattr(module.joblib, "dump", _dump_to_path)

    saved_path = module.save_pipeline(cast(Pipeline, SimpleNamespace()), target_dir)

    assert saved_path == target_dir / "pipeline.joblib"
    assert target_dir.exists()


def test_save_pipeline_preserves_existing_artifact_when_dump_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep existing pipeline artifact unchanged when temp-file dump fails."""
    pipeline_file = tmp_path / "pipeline.joblib"
    pipeline_file.write_text("stable-pipeline", encoding="utf-8")

    def _failing_dump(*_args: object, **_kwargs: object) -> None:
        raise OSError("dump failed")

    monkeypatch.setattr(module.joblib, "dump", _failing_dump)

    with pytest.raises(PersistenceError, match="Failed to save pipeline"):
        module.save_pipeline(cast(Pipeline, SimpleNamespace()), tmp_path)

    assert pipeline_file.read_text(encoding="utf-8") == "stable-pipeline"


def test_save_pipeline_cleans_temp_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remove temporary pipeline artifact when atomic replace operation fails."""
    def _dump_to_path(_pipeline: object, pipeline_file: Path) -> None:
        pipeline_file.write_text("temp", encoding="utf-8")

    monkeypatch.setattr(module.joblib, "dump", _dump_to_path)

    captured_temp_path: dict[str, Path] = {}

    def _failing_replace(src: str | Path, dst: str | Path) -> None:
        _ = dst
        captured_temp_path["path"] = Path(src)
        raise OSError("replace blocked")

    monkeypatch.setattr(module.os, "replace", _failing_replace)

    with pytest.raises(PersistenceError, match="Failed to save pipeline"):
        module.save_pipeline(cast(Pipeline, SimpleNamespace()), tmp_path)

    assert "path" in captured_temp_path
    assert not captured_temp_path["path"].exists()
