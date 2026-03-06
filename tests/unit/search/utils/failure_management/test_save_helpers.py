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

    def _failing_open(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", _failing_open)

    with pytest.raises(PersistenceError, match="Failed to save best broad params"):
        save_broad(
            broad_result={"best_params": {"Model__depth": 6}},
            best_params_1={"Model__depth": 6},
            tgt_file=tgt_file,
        )


def test_save_narrow_wraps_write_failures_as_persistence_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Wrap file-open/write failures with narrow-save context in `PersistenceError`."""
    tgt_file = tmp_path / "narrow_info.json"

    def _failing_open(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "open", _failing_open)

    with pytest.raises(PersistenceError, match="Failed to save narrow search done marker"):
        save_narrow(
            narrow_result={"best_params": {"Model__depth": 7}},
            best_params={"Model__depth": 7},
            tgt_file=tgt_file,
        )
