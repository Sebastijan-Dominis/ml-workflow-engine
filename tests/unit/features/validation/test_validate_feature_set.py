"""Unit tests for feature-set validation and hash-integrity checks."""

from pathlib import Path

import pandas as pd
import pytest
from ml.exceptions import DataError
from ml.features.validation.validate_feature_set import validate_feature_set

pytestmark = pytest.mark.unit


def _metadata(*, schema: str = "schema-ok", in_memory: str = "mem-ok", file_hash: str = "file-ok") -> dict:
    """Build metadata payloads used by feature-set validation tests."""
    return {
        "feature_schema_hash": schema,
        "in_memory_hash": in_memory,
        "file_hash": file_hash,
    }


def test_validate_feature_set_raises_when_row_id_column_missing() -> None:
    """Reject feature sets that do not include the mandatory `row_id` column."""
    feature_set = pd.DataFrame({"feature_a": [1, 2]})

    with pytest.raises(DataError, match="missing required 'row_id' column"):
        validate_feature_set(
            feature_set,
            metadata=_metadata(),
            file_path=Path("a/b/features.parquet"),
            strict=True,
        )


def test_validate_feature_set_raises_on_schema_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail validation when computed schema hash differs from metadata contract."""
    feature_set = pd.DataFrame({"row_id": [1, 2], "feature_a": [10.0, 20.0]})

    import pandas as _pd
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.hash_feature_schema",
        lambda df: "schema-actual",
    )
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.load_feature_set_schemas",
        lambda features_path, file_path: (_pd.DataFrame(), _pd.DataFrame()),
    )

    with pytest.raises(DataError, match="Feature schema hash mismatch"):
        validate_feature_set(
            feature_set,
            metadata=_metadata(schema="schema-expected"),
            file_path=Path("a/b/features.parquet"),
            strict=True,
        )


def test_validate_feature_set_strict_false_skips_in_memory_and_file_hash_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow validation to pass with strict mode disabled after schema check succeeds."""
    feature_set = pd.DataFrame({"row_id": [1, 2], "feature_a": [10.0, 20.0]})

    import pandas as _pd
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.hash_feature_schema",
        lambda df: "schema-ok",
    )
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.load_feature_set_schemas",
        lambda features_path, file_path: (_pd.DataFrame(), _pd.DataFrame()),
    )

    def _should_not_be_called(*args: object, **kwargs: object) -> str:
        raise AssertionError("Strict=False should skip in-memory and file hash checks")

    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.hash_dataframe_content",
        _should_not_be_called,
    )
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.hash_file",
        _should_not_be_called,
    )

    validate_feature_set(
        feature_set,
        metadata=_metadata(schema="schema-ok", in_memory="ignored", file_hash="ignored"),
        file_path=Path("a/b/features.parquet"),
        strict=False,
    )


def test_validate_feature_set_logs_warning_for_in_memory_hash_mismatch_only(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn on in-memory hash mismatch but continue when file hash still matches."""
    feature_set = pd.DataFrame({"row_id": [1, 2], "feature_a": [10.0, 20.0]})


    import pandas as _pd
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.hash_feature_schema",
        lambda df: "schema-ok",
    )
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.load_feature_set_schemas",
        lambda features_path, file_path: (_pd.DataFrame(), _pd.DataFrame()),
    )
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.hash_dataframe_content",
        lambda df: "mem-actual",
    )
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.hash_file",
        lambda path: "file-ok",
    )

    with caplog.at_level("WARNING"):
        validate_feature_set(
            feature_set,
            metadata=_metadata(schema="schema-ok", in_memory="mem-expected", file_hash="file-ok"),
            file_path=Path("a/b/features.parquet"),
            strict=True,
        )

    assert "In-memory feature hash mismatch" in caplog.text


def test_validate_feature_set_raises_on_file_hash_mismatch_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise DataError when strict validation detects persisted-file hash drift."""
    feature_set = pd.DataFrame({"row_id": [1, 2], "feature_a": [10.0, 20.0]})


    import pandas as _pd
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.hash_feature_schema",
        lambda df: "schema-ok",
    )
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.load_feature_set_schemas",
        lambda features_path, file_path: (_pd.DataFrame(), _pd.DataFrame()),
    )
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.hash_dataframe_content",
        lambda df: "mem-ok",
    )
    monkeypatch.setattr(
        "ml.features.validation.validate_feature_set.hash_file",
        lambda path: "file-actual",
    )

    with pytest.raises(DataError, match="File hash mismatch"):
        validate_feature_set(
            feature_set,
            metadata=_metadata(schema="schema-ok", in_memory="mem-ok", file_hash="file-expected"),
            file_path=Path("a/b/features.parquet"),
            strict=True,
        )
