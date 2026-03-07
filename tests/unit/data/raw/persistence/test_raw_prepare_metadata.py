"""Unit tests for raw snapshot metadata preparation helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.data.raw.persistence import prepare_metadata as module
from ml.exceptions import PersistenceError
from ml.metadata.schemas.data.raw import RawSnapshotMetadata

pytestmark = pytest.mark.unit


class _FixedDateTime:
    """Datetime stub returning a deterministic ISO timestamp."""

    @classmethod
    def now(cls) -> Any:
        """Return object with stable ``isoformat`` for deterministic tests."""
        return SimpleNamespace(isoformat=lambda: "2026-03-07T12:00:00")


def test_prepare_raw_metadata_builds_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Build metadata payload and pass it to schema validation unchanged."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    data_path = tmp_path / "hotel_bookings.csv"
    args = SimpleNamespace(data="hotel_bookings", version="v1", owner="data-team")

    monkeypatch.setattr(module, "datetime", _FixedDateTime)
    monkeypatch.setattr(module, "hash_data", lambda _path: "hash-123")
    monkeypatch.setattr(module, "get_memory_usage", lambda _df: 1.25)

    captured: dict[str, Any] = {}
    expected_metadata = SimpleNamespace()

    def _validate(raw: dict[str, Any]) -> RawSnapshotMetadata:
        captured["raw"] = raw
        return cast(RawSnapshotMetadata, expected_metadata)

    monkeypatch.setattr(module, "validate_raw_snapshot_metadata", _validate)

    result = module.prepare_metadata(
        df,
        args=args,
        data_path=data_path,
        raw_run_id="raw_001",
        data_format="csv",
        data_suffix="raw/hotel_bookings.csv",
    )

    assert result is expected_metadata
    raw = captured["raw"]
    assert raw["data"]["name"] == "hotel_bookings"
    assert raw["data"]["version"] == "v1"
    assert raw["data"]["hash"] == "hash-123"
    assert raw["rows"] == 2
    assert raw["columns"]["count"] == 2
    assert raw["created_at"] == "2026-03-07T12:00:00"
    assert raw["owner"] == "data-team"
    assert raw["memory_usage_mb"] == 1.25
    assert raw["raw_run_id"] == "raw_001"


def test_prepare_raw_metadata_wraps_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Wrap metadata preparation failures as ``PersistenceError``."""
    df = pd.DataFrame({"a": [1]})
    args = SimpleNamespace(data="hotel_bookings", version="v1", owner="data-team")

    monkeypatch.setattr(module, "datetime", _FixedDateTime)
    monkeypatch.setattr(module, "hash_data", lambda _path: "hash-123")
    monkeypatch.setattr(module, "get_memory_usage", lambda _df: 0.1)
    monkeypatch.setattr(
        module,
        "validate_raw_snapshot_metadata",
        lambda _raw: (_ for _ in ()).throw(ValueError("bad schema")),
    )

    with pytest.raises(PersistenceError, match="Failed to prepare metadata for raw data"):
        module.prepare_metadata(
            df,
            args=args,
            data_path=tmp_path / "hotel_bookings.csv",
            raw_run_id="raw_002",
            data_format="csv",
            data_suffix="raw/hotel_bookings.csv",
        )
