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
    dumped_path = Path(captured["path"])
    assert dumped_path.parent == tmp_path
    assert dumped_path.name.startswith("model.")
    assert dumped_path.suffixes[-2:] == [".joblib", ".tmp"]
    assert saved_path.exists()


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


def test_save_model_creates_missing_parent_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Create artifact directory before dumping model when target path does not exist."""
    target_dir = tmp_path / "nested" / "artifacts"

    monkeypatch.setattr(module.joblib, "dump", lambda model, model_file: Path(model_file).write_text("model"))

    saved_path = module.save_model(SimpleNamespace(), target_dir)

    assert saved_path == target_dir / "model.joblib"
    assert target_dir.exists()


def test_save_model_preserves_existing_artifact_when_dump_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep existing model artifact intact when temp-file dump fails."""
    model_file = tmp_path / "model.joblib"
    model_file.write_text("stable-model", encoding="utf-8")

    def _failing_dump(*_args: object, **_kwargs: object) -> None:
        raise OSError("dump failed")

    monkeypatch.setattr(module.joblib, "dump", _failing_dump)

    with pytest.raises(PersistenceError, match="Failed to save model"):
        module.save_model(SimpleNamespace(), tmp_path)

    assert model_file.read_text(encoding="utf-8") == "stable-model"


def test_save_model_cleans_temp_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remove temporary model artifact when atomic replace operation fails."""
    monkeypatch.setattr(module.joblib, "dump", lambda model, model_file: Path(model_file).write_text("temp"))

    captured_temp_path: dict[str, Path] = {}

    def _failing_replace(src: str | Path, dst: str | Path) -> None:
        _ = dst
        captured_temp_path["path"] = Path(src)
        raise OSError("replace blocked")

    monkeypatch.setattr(module.os, "replace", _failing_replace)

    with pytest.raises(PersistenceError, match="Failed to save model"):
        module.save_model(SimpleNamespace(), tmp_path)

    assert "path" in captured_temp_path
    assert not captured_temp_path["path"].exists()
