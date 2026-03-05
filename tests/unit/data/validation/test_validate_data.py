"""Unit tests for dataset hash validation against metadata expectations."""

from pathlib import Path

import pytest
from ml.data.validation.validate_data import validate_data
from ml.exceptions import UserError

pytestmark = pytest.mark.unit


def test_validate_data_returns_empty_and_warns_when_metadata_hash_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Skip integrity checks when metadata does not include expected data hash."""
    metadata = {"data": {}}

    with caplog.at_level("WARNING"):
        result = validate_data(data_path=Path("dummy.parquet"), metadata=metadata)

    assert result == ""
    assert "No data hash found in metadata" in caplog.text


def test_validate_data_returns_actual_hash_when_expected_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return computed hash when metadata hash matches persisted data hash."""
    metadata = {"data": {"hash": "abc123"}}
    monkeypatch.setattr("ml.data.validation.validate_data.hash_data", lambda path: "abc123")

    result = validate_data(data_path=Path("dummy.parquet"), metadata=metadata)

    assert result == "abc123"


def test_validate_data_raises_when_expected_hash_mismatches_actual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject datasets whose computed hash differs from metadata expectation."""
    metadata = {"data": {"hash": "expected"}}
    monkeypatch.setattr("ml.data.validation.validate_data.hash_data", lambda path: "actual")

    with pytest.raises(UserError, match="Data hash mismatch"):
        validate_data(data_path=Path("dummy.parquet"), metadata=metadata)
