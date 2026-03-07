"""Unit tests for explainability metrics CSV persistence helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from ml.exceptions import PersistenceError
from ml.runners.explainability.persistence.save_metrics_csv import save_metrics_csv

pytestmark = pytest.mark.unit


def test_save_metrics_csv_creates_parent_directory_and_writes_expected_content(
    tmp_path: Path,
) -> None:
    """Create missing parent folders and persist dataframe as CSV with no index."""
    target_file = tmp_path / "nested" / "metrics" / "feature_importances.csv"
    df = pd.DataFrame(
        {
            "feature": ["adr", "lead_time"],
            "importance": [0.42, 0.37],
        }
    )

    save_metrics_csv(df, target_file=target_file, name="Feature importances")

    assert target_file.exists()
    loaded = pd.read_csv(target_file)
    assert list(loaded.columns) == ["feature", "importance"]
    assert loaded.to_dict(orient="records") == [
        {"feature": "adr", "importance": 0.42},
        {"feature": "lead_time", "importance": 0.37},
    ]


def test_save_metrics_csv_logs_success_message(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Emit informative success log containing table name and destination path."""
    target_file = tmp_path / "shap.csv"
    df = pd.DataFrame({"feature": ["total_stay"], "importance": [0.9]})

    with caplog.at_level("INFO", logger="ml.runners.explainability.persistence.save_metrics_csv"):
        save_metrics_csv(df, target_file=target_file, name="SHAP importances")

    assert f"SHAP importances successfully saved to {target_file}." in caplog.text


def test_save_metrics_csv_wraps_to_csv_errors_with_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Wrap write failures with table/path context and preserve original cause."""
    target_file = tmp_path / "broken" / "out.csv"
    df = pd.DataFrame({"feature": ["adr"], "importance": [0.1]})

    def _fail_to_csv(path: Path, *, index: bool) -> None:
        _ = (path, index)
        raise OSError("permission denied")

    monkeypatch.setattr(df, "to_csv", _fail_to_csv)

    with caplog.at_level("ERROR", logger="ml.runners.explainability.persistence.save_metrics_csv"), pytest.raises(
        PersistenceError,
        match="Failed to save Feature importances",
    ) as exc_info:
        save_metrics_csv(df, target_file=target_file, name="Feature importances")

    assert isinstance(exc_info.value.__cause__, OSError)
    assert "Failed to save Feature importances" in caplog.text


def test_save_metrics_csv_preserves_existing_file_when_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep existing CSV intact when temporary serialization fails."""
    target_file = tmp_path / "existing.csv"
    target_file.write_text("feature,importance\nexisting,1.0\n", encoding="utf-8")
    df = pd.DataFrame({"feature": ["adr"], "importance": [0.1]})

    def _fail_to_csv(path: Path, *, index: bool) -> None:
        _ = (path, index)
        raise OSError("disk full")

    monkeypatch.setattr(df, "to_csv", _fail_to_csv)

    with pytest.raises(PersistenceError, match="Failed to save Feature importances"):
        save_metrics_csv(df, target_file=target_file, name="Feature importances")

    assert target_file.read_text(encoding="utf-8") == "feature,importance\nexisting,1.0\n"


def test_save_metrics_csv_cleans_temp_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Delete temporary CSV file when atomic replace operation fails."""
    target_file = tmp_path / "shap.csv"
    df = pd.DataFrame({"feature": ["x"], "importance": [0.2]})

    captured_temp_path: dict[str, Path] = {}

    def _failing_replace(src: str | Path, dst: str | Path) -> None:
        _ = dst
        captured_temp_path["path"] = Path(src)
        raise OSError("replace blocked")

    monkeypatch.setattr("ml.runners.explainability.persistence.save_metrics_csv.os.replace", _failing_replace)

    with pytest.raises(PersistenceError, match="Failed to save SHAP importances"):
        save_metrics_csv(df, target_file=target_file, name="SHAP importances")

    assert "path" in captured_temp_path
    assert not captured_temp_path["path"].exists()
