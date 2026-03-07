"""Unit tests for interim dataset metadata preparation helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.data.config.schemas.interim import InterimConfig
from ml.data.interim.persistence import prepare_metadata as module
from ml.metadata.schemas.data.interim import InterimDatasetMetadata

pytestmark = pytest.mark.unit


def _interim_config_stub() -> InterimConfig:
    """Build minimal ``InterimConfig``-compatible object for metadata tests."""
    return cast(
        InterimConfig,
        SimpleNamespace(
            data=SimpleNamespace(
                name="hotel_bookings",
                version="v2",
                output=SimpleNamespace(path_suffix="interim/hotel_bookings", format="parquet", compression="snappy"),
            ),
            raw_data_version="v1",
        ),
    )


def test_prepare_interim_metadata_builds_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Build interim metadata payload and pass it through validation."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    cfg = _interim_config_stub()

    monkeypatch.setattr(module, "hash_data", lambda _path: "hash-123")
    monkeypatch.setattr(module, "compute_data_config_hash", lambda _cfg: "cfg-hash")
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260307T120000")
    monkeypatch.setattr(module.time, "perf_counter", lambda: 15.0)

    captured: dict[str, Any] = {}
    expected_metadata = SimpleNamespace()

    def _validate(raw: dict[str, Any]) -> InterimDatasetMetadata:
        captured["raw"] = raw
        return cast(InterimDatasetMetadata, expected_metadata)

    monkeypatch.setattr(module, "validate_interim_dataset_metadata", _validate)

    source_data_path = tmp_path / "raw" / "snapshot_v1" / "raw.parquet"
    result = module.prepare_metadata(
        df,
        config=cfg,
        start_time=10.0,
        data_path=tmp_path / "interim.parquet",
        source_data_path=source_data_path,
        source_data_format="parquet",
        owner="data-team",
        memory_info={"before_mb": 100.0, "after_mb": 110.0, "delta_mb": 10.0},
        interim_run_id="interim_001",
    )

    assert result is expected_metadata
    raw = captured["raw"]
    assert raw["interim_run_id"] == "interim_001"
    assert raw["source_data"]["snapshot_id"] == "snapshot_v1"
    assert raw["data"]["hash"] == "hash-123"
    assert raw["config_hash"] == "cfg-hash"
    assert raw["created_at"] == "20260307T120000"
    assert raw["duration"] == pytest.approx(5.0)
    assert raw["rows"] == 2
    assert raw["columns"]["count"] == 2
