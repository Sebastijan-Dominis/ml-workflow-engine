"""Unit tests for metadata persistence helper."""

import json
from pathlib import Path
from typing import Any

import pytest
from ml.exceptions import PersistenceError
from ml.io.persistence import save_metadata as module

pytestmark = pytest.mark.unit


def test_save_metadata_writes_metadata_json(tmp_path: Path) -> None:
    """Write metadata payload to `metadata.json` in the target directory."""
    target_dir = tmp_path / "meta"
    metadata = {"run_id": "abc", "score": 0.91}

    module.save_metadata(metadata, target_dir=target_dir)

    saved = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved == metadata


def test_save_metadata_rejects_existing_file_when_overwrite_disabled(tmp_path: Path) -> None:
    """Raise `PersistenceError` when target file exists and overwrite is disabled."""
    target_dir = tmp_path / "meta"
    target_dir.mkdir(parents=True)
    (target_dir / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PersistenceError, match="already exists"):
        module.save_metadata({"new": 1}, target_dir=target_dir, overwrite_existing=False)


def test_save_metadata_overwrites_existing_file_when_enabled(tmp_path: Path) -> None:
    """Overwrite existing `metadata.json` when overwrite is enabled."""
    target_dir = tmp_path / "meta"
    target_dir.mkdir(parents=True)
    (target_dir / "metadata.json").write_text('{"old": 1}', encoding="utf-8")

    module.save_metadata({"new": 2}, target_dir=target_dir, overwrite_existing=True)

    saved = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved == {"new": 2}


def test_save_metadata_preserves_existing_file_when_serialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep original metadata intact when temporary-file serialization raises an error."""
    target_dir = tmp_path / "meta"
    target_dir.mkdir(parents=True)
    metadata_file = target_dir / "metadata.json"
    metadata_file.write_text('{"stable": true}', encoding="utf-8")

    def _failing_dump(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk write failed")

    monkeypatch.setattr(module.json, "dump", _failing_dump)

    with pytest.raises(PersistenceError, match="Failed to save metadata"):
        module.save_metadata({"new": 3}, target_dir=target_dir, overwrite_existing=True)

    assert json.loads(metadata_file.read_text(encoding="utf-8")) == {"stable": True}


def test_save_metadata_cleans_temp_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remove temporary metadata file when atomic replace fails unexpectedly."""
    target_dir = tmp_path / "meta"

    captured_temp_path: dict[str, Path] = {}

    def _failing_replace(src: str | Path, dst: str | Path) -> None:
        captured_temp_path["path"] = Path(src)
        raise OSError("replace blocked")

    monkeypatch.setattr(module.os, "replace", _failing_replace)

    with pytest.raises(PersistenceError, match="Failed to save metadata"):
        module.save_metadata({"new": 4}, target_dir=target_dir, overwrite_existing=True)

    assert "path" in captured_temp_path
    assert not captured_temp_path["path"].exists()
