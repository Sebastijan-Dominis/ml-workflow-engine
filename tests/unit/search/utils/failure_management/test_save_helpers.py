"""Unit tests for failure-management save helper utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ml.exceptions import PersistenceError
from ml.search.utils.failure_management.save_broad import save_broad
from ml.search.utils.failure_management.save_narrow import save_narrow

pytestmark = pytest.mark.unit


def test_save_broad_writes_expected_json_payload(tmp_path: Path) -> None:
    """Persist broad result and best params under expected top-level keys."""
    tgt_file = tmp_path / "broad_info.json"

    save_broad(
        broad_result={"best_params": {"Model__depth": 6}, "best_score": 0.8},
        best_params_1={"Model__depth": 6},
        tgt_file=tgt_file,
    )

    saved = json.loads(tgt_file.read_text(encoding="utf-8"))
    assert saved == {
        "broad_result": {"best_params": {"Model__depth": 6}, "best_score": 0.8},
        "best_params_1": {"Model__depth": 6},
    }


def test_save_narrow_writes_expected_json_payload(tmp_path: Path) -> None:
    """Persist narrow result and best params under expected top-level keys."""
    tgt_file = tmp_path / "narrow_info.json"

    save_narrow(
        narrow_result={"best_params": {"Model__depth": 7}, "best_score": 0.82},
        best_params={"Model__depth": 7},
        tgt_file=tgt_file,
    )

    saved = json.loads(tgt_file.read_text(encoding="utf-8"))
    assert saved == {
        "narrow_result": {"best_params": {"Model__depth": 7}, "best_score": 0.82},
        "best_params": {"Model__depth": 7},
    }


def test_save_broad_wraps_write_failures_as_persistence_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Wrap file-open/write failures with broad-save context in `PersistenceError`."""
    tgt_file = tmp_path / "broad_info.json"

    def _failing_dump(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("ml.search.utils.failure_management.save_broad.json.dump", _failing_dump)

    with pytest.raises(PersistenceError, match="Failed to save best broad params"):
        save_broad(
            broad_result={"best_params": {"Model__depth": 6}},
            best_params_1={"Model__depth": 6},
            tgt_file=tgt_file,
        )


def test_save_narrow_wraps_write_failures_as_persistence_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Wrap file-open/write failures with narrow-save context in `PersistenceError`."""
    tgt_file = tmp_path / "narrow_info.json"

    def _failing_dump(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("ml.search.utils.failure_management.save_narrow.json.dump", _failing_dump)

    with pytest.raises(PersistenceError, match="Failed to save narrow search done marker"):
        save_narrow(
            narrow_result={"best_params": {"Model__depth": 7}},
            best_params={"Model__depth": 7},
            tgt_file=tgt_file,
        )


def test_save_broad_preserves_existing_file_when_dump_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep existing broad marker content unchanged when temporary write fails."""
    tgt_file = tmp_path / "broad_info.json"
    tgt_file.write_text('{"stable": true}', encoding="utf-8")

    monkeypatch.setattr("ml.search.utils.failure_management.save_broad.json.dump", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(PersistenceError, match="Failed to save best broad params"):
        save_broad(
            broad_result={"best_params": {"Model__depth": 6}},
            best_params_1={"Model__depth": 6},
            tgt_file=tgt_file,
        )

    assert json.loads(tgt_file.read_text(encoding="utf-8")) == {"stable": True}


def test_save_narrow_preserves_existing_file_when_dump_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep existing narrow marker content unchanged when temporary write fails."""
    tgt_file = tmp_path / "narrow_info.json"
    tgt_file.write_text('{"stable": true}', encoding="utf-8")

    monkeypatch.setattr("ml.search.utils.failure_management.save_narrow.json.dump", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(PersistenceError, match="Failed to save narrow search done marker"):
        save_narrow(
            narrow_result={"best_params": {"Model__depth": 7}},
            best_params={"Model__depth": 7},
            tgt_file=tgt_file,
        )

    assert json.loads(tgt_file.read_text(encoding="utf-8")) == {"stable": True}


def test_save_broad_cleans_temp_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Delete temporary broad marker file when atomic replace fails."""
    tgt_file = tmp_path / "broad_info.json"

    captured_temp_path: dict[str, Path] = {}

    def _failing_replace(src: str | Path, dst: str | Path) -> None:
        _ = dst
        captured_temp_path["path"] = Path(src)
        raise OSError("replace blocked")

    monkeypatch.setattr("ml.search.utils.failure_management.save_broad.os.replace", _failing_replace)

    with pytest.raises(PersistenceError, match="Failed to save best broad params"):
        save_broad(
            broad_result={"best_params": {"Model__depth": 6}},
            best_params_1={"Model__depth": 6},
            tgt_file=tgt_file,
        )

    assert "path" in captured_temp_path
    assert not captured_temp_path["path"].exists()


def test_save_narrow_cleans_temp_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Delete temporary narrow marker file when atomic replace fails."""
    tgt_file = tmp_path / "narrow_info.json"

    captured_temp_path: dict[str, Path] = {}

    def _failing_replace(src: str | Path, dst: str | Path) -> None:
        _ = dst
        captured_temp_path["path"] = Path(src)
        raise OSError("replace blocked")

    monkeypatch.setattr("ml.search.utils.failure_management.save_narrow.os.replace", _failing_replace)

    with pytest.raises(PersistenceError, match="Failed to save narrow search done marker"):
        save_narrow(
            narrow_result={"best_params": {"Model__depth": 7}},
            best_params={"Model__depth": 7},
            tgt_file=tgt_file,
        )

    assert "path" in captured_temp_path
    assert not captured_temp_path["path"].exists()
