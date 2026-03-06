"""Unit tests for evaluation prediction artifact persistence helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from ml.exceptions import PersistenceError
from ml.runners.evaluation.models.predictions import PredictionArtifacts
from ml.runners.evaluation.persistence.save_predictions import save_predictions

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

    def _fake_to_parquet(self: pd.DataFrame, path: Path, *args: object, **kwargs: object) -> None:
        _ = (self, args, kwargs)
        written_paths.append(path)
        path.write_bytes(b"parquet-bytes")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)

    target_dir = tmp_path / "eval" / "run-1"
    result = save_predictions(_prediction_artifacts(), target_dir)

    expected_train = target_dir / "predictions_train.parquet"
    expected_val = target_dir / "predictions_val.parquet"
    expected_test = target_dir / "predictions_test.parquet"

    assert target_dir.exists()
    assert written_paths == [expected_train, expected_val, expected_test]
    assert result.train_predictions_path == str(expected_train)
    assert result.val_predictions_path == str(expected_val)
    assert result.test_predictions_path == str(expected_test)


def test_save_predictions_wraps_parquet_failures_as_persistence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap per-split write failures with target-dir context for debugging."""

    def _failing_to_parquet(self: pd.DataFrame, path: Path, *args: object, **kwargs: object) -> None:
        _ = (self, args, kwargs)
        if path.name == "predictions_val.parquet":
            raise OSError("disk full")
        path.write_bytes(b"ok")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _failing_to_parquet)

    target_dir = tmp_path / "eval" / "run-2"

    with pytest.raises(PersistenceError, match="Failed to save predictions") as exc_info:
        save_predictions(_prediction_artifacts(), target_dir)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(target_dir) in str(exc_info.value)
    assert target_dir.exists()
