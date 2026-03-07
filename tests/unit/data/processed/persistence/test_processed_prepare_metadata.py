"""Unit tests for processed dataset metadata preparation helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.data.config.schemas.processed import ProcessedConfig
from ml.data.processed.persistence import prepare_metadata as module
from ml.metadata.schemas.data.processed import ProcessedDatasetMetadata

pytestmark = pytest.mark.unit


def _processed_config_stub() -> ProcessedConfig:
    """Build minimal ``ProcessedConfig``-compatible object for metadata tests."""
    return cast(
        ProcessedConfig,
        SimpleNamespace(
            data=SimpleNamespace(
                name="hotel_bookings",
                version="v3",
                output=SimpleNamespace(path_suffix="processed/hotel_bookings", format="parquet", compression="snappy"),
            ),
        ),
    )


def test_prepare_processed_metadata_includes_row_id_info_when_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Include ``row_id_info`` in metadata payload when supplied by processing stage."""
    df = pd.DataFrame({"row_id": ["r1", "r2"], "a": [1, 2]})
    cfg = _processed_config_stub()

    monkeypatch.setattr(module, "hash_data", lambda _path: "hash-xyz")
    monkeypatch.setattr(module, "compute_data_config_hash", lambda _cfg: "cfg-hash")
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260307T121500")
    monkeypatch.setattr(module.time, "perf_counter", lambda: 42.0)

    captured: dict[str, Any] = {}
    expected_metadata = SimpleNamespace()

    def _validate(raw: dict[str, Any]) -> ProcessedDatasetMetadata:
        captured["raw"] = raw
        return cast(ProcessedDatasetMetadata, expected_metadata)

    monkeypatch.setattr(module, "validate_processed_dataset_metadata", _validate)

    source_data_path = tmp_path / "interim" / "snapshot_v2" / "interim.parquet"
    result = module.prepare_metadata(
        df,
        config=cfg,
        start_time=40.0,
        data_path=tmp_path / "processed.parquet",
        source_data_path=source_data_path,
        source_data_format="parquet",
        source_data_version="v2",
        owner="data-team",
        memory_info={"before_mb": 120.0, "after_mb": 130.0, "delta_mb": 10.0},
        processed_run_id="processed_001",
        row_id_info=cast(Any, {"method": "fingerprint", "version": "v1"}),
    )

    assert result is expected_metadata
    raw = captured["raw"]
    assert raw["processed_run_id"] == "processed_001"
    assert raw["source_data"]["snapshot_id"] == "snapshot_v2"
    assert raw["data"]["hash"] == "hash-xyz"
    assert raw["config_hash"] == "cfg-hash"
    assert raw["duration"] == pytest.approx(2.0)
    assert raw["row_id_info"] == {"method": "fingerprint", "version": "v1"}


def test_prepare_processed_metadata_omits_row_id_info_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omit ``row_id_info`` from metadata payload when no row-id trace is supplied."""
    df = pd.DataFrame({"row_id": ["r1"], "a": [1]})
    cfg = _processed_config_stub()

    monkeypatch.setattr(module, "hash_data", lambda _path: "hash-xyz")
    monkeypatch.setattr(module, "compute_data_config_hash", lambda _cfg: "cfg-hash")
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260307T121500")
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.0)

    captured: dict[str, Any] = {}
    expected_metadata = SimpleNamespace()

    def _validate(raw: dict[str, Any]) -> ProcessedDatasetMetadata:
        captured["raw"] = raw
        return cast(ProcessedDatasetMetadata, expected_metadata)

    monkeypatch.setattr(module, "validate_processed_dataset_metadata", _validate)

    result = module.prepare_metadata(
        df,
        config=cfg,
        start_time=9.0,
        data_path=tmp_path / "processed.parquet",
        source_data_path=tmp_path / "interim" / "snapshot_v2" / "interim.parquet",
        source_data_format="parquet",
        source_data_version="v2",
        owner="data-team",
        memory_info={"before_mb": 120.0, "after_mb": 130.0, "delta_mb": 10.0},
        processed_run_id="processed_002",
        row_id_info=None,
    )

    assert result is expected_metadata
    assert "row_id_info" not in captured["raw"]
