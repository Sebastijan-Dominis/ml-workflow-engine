"""Unit tests for evaluation prediction artifact persistence helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from ml.exceptions import PersistenceError
from ml.runners.evaluation.models.predictions import PredictionArtifacts
from ml.runners.evaluation.persistence import save_predictions as module

pytestmark = pytest.mark.unit


def _prediction_artifacts() -> PredictionArtifacts:
    """Build small deterministic per-split prediction dataframes."""
    return PredictionArtifacts(
        train=pd.DataFrame({"row_id": [1], "y_pred": [0.1]}),
        val=pd.DataFrame({"row_id": [2], "y_pred": [0.2]}),
        test=pd.DataFrame({"row_id": [3], "y_pred": [0.3]}),
    )


def test_save_predictions_writes_all_split_parquets_and_returns_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist train/val/test artifacts and return expected path model values."""
    written_paths: list[Path] = []

    def _fake_to_parquet(self: pd.DataFrame, path: str | Path, *args: object, **kwargs: object) -> None:
        _ = (self, args, kwargs)
        out_path = Path(path)
        written_paths.append(out_path)
        out_path.write_bytes(b"parquet-bytes")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)

    target_dir = tmp_path / "eval" / "run-1"
    result = module.save_predictions(_prediction_artifacts(), target_dir)

    expected_train = target_dir / "predictions_train.parquet"
    expected_val = target_dir / "predictions_val.parquet"
    expected_test = target_dir / "predictions_test.parquet"

    assert target_dir.exists()
    assert len(written_paths) == 3
    assert all(path.parent == target_dir for path in written_paths)
    assert all(path.name.endswith(".parquet.tmp") for path in written_paths)
    assert result.train_predictions_path == Path(expected_train).as_posix()
    assert result.val_predictions_path == Path(expected_val).as_posix()
    assert result.test_predictions_path == Path(expected_test).as_posix()


def test_save_predictions_wraps_parquet_failures_as_persistence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap per-split write failures with target-dir context for debugging."""

    call_count = {"value": 0}

    def _failing_to_parquet(self: pd.DataFrame, path: str | Path, *args: object, **kwargs: object) -> None:
        _ = (self, args, kwargs)
        call_count["value"] += 1
        out_path = Path(path)
        if call_count["value"] == 2:
            raise OSError("disk full")
        out_path.write_bytes(b"ok")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _failing_to_parquet)

    target_dir = tmp_path / "eval" / "run-2"

    with pytest.raises(PersistenceError, match="Failed to save predictions") as exc_info:
        module.save_predictions(_prediction_artifacts(), target_dir)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(target_dir) in str(exc_info.value)
    assert target_dir.exists()


def test_save_predictions_preserves_existing_val_file_when_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep pre-existing validation predictions file unchanged when write fails mid-run."""
    target_dir = tmp_path / "eval" / "run-3"
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_val = target_dir / "predictions_val.parquet"
    existing_val.write_bytes(b"stable")

    call_count = {"value": 0}

    def _failing_to_parquet(self: pd.DataFrame, path: str | Path, *args: object, **kwargs: object) -> None:
        _ = (self, args, kwargs)
        call_count["value"] += 1
        out_path = Path(path)
        if call_count["value"] == 2:
            raise OSError("val write failed")
        out_path.write_bytes(b"ok")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _failing_to_parquet)

    with pytest.raises(PersistenceError, match="Failed to save predictions"):
        module.save_predictions(_prediction_artifacts(), target_dir)

    assert existing_val.read_bytes() == b"stable"


def test_save_predictions_cleans_temp_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delete temporary parquet file when atomic replace operation fails."""
    target_dir = tmp_path / "eval" / "run-4"

    def _fake_to_parquet(self: pd.DataFrame, path: str | Path, *args: object, **kwargs: object) -> None:
        _ = (self, args, kwargs)
        Path(path).write_bytes(b"ok")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)

    captured_temp_path: dict[str, Path] = {}

    def _failing_replace(src: str | Path, dst: str | Path) -> None:
        _ = dst
        captured_temp_path["path"] = Path(src)
        raise OSError("replace blocked")

    monkeypatch.setattr(module.os, "replace", _failing_replace)

    with pytest.raises(PersistenceError, match="Failed to save predictions"):
        module.save_predictions(_prediction_artifacts(), target_dir)

    assert "path" in captured_temp_path
    assert not captured_temp_path["path"].exists()
