"""Unit tests for interim-dataset build CLI behavior."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from pipelines.data import build_interim_dataset as module

pytestmark = pytest.mark.unit


def test_parse_args_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse default raw snapshot, logging level, and owner values."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["build_interim_dataset", "--data", "hotel_bookings", "--version", "v1"],
    )

    args = module.parse_args()

    assert args.data == "hotel_bookings"
    assert args.version == "v1"
    assert args.raw_snapshot_id == "latest"
    assert args.logging_level == "INFO"
    assert args.owner == "Sebastijan"


def test_main_happy_path_runs_pipeline_and_persists_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Execute full interim pipeline sequence and persist metadata successfully."""
    monkeypatch.chdir(tmp_path)

    args = Namespace(
        data="hotel_bookings",
        version="v1",
        raw_snapshot_id="latest",
        logging_level="INFO",
        owner="Sebastijan",
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T150000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        module,
        "setup_logging",
        lambda path, level: captured.setdefault("log", (path, level)),
    )

    raw_snapshot_path = tmp_path / "data" / "raw" / "hotel_bookings" / "v_raw" / "run_001"
    raw_snapshot_path.mkdir(parents=True)

    config = SimpleNamespace(
        raw_data_version="v_raw",
        cleaning=SimpleNamespace(),
        data_schema=SimpleNamespace(),
        drop_missing_ints=False,
        invariants=SimpleNamespace(),
        min_rows=1,
        drop_duplicates=True,
    )

    monkeypatch.setattr(module, "load_yaml", lambda _path: {"config": "raw"})
    monkeypatch.setattr(module, "validate_config", lambda raw, type: config)
    monkeypatch.setattr(module, "get_snapshot_path", lambda _sid, _parent: raw_snapshot_path)

    raw_metadata = {"data": {"suffix": "data.csv"}}
    monkeypatch.setattr(module, "load_json", lambda _path: raw_metadata)
    monkeypatch.setattr(module, "get_data_suffix_and_format", lambda meta, location: ("data.csv", "csv"))
    monkeypatch.setattr(module, "validate_data", lambda data_path, metadata: None)

    df_in = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
    monkeypatch.setattr(module, "read_data", lambda fmt, path: df_in)
    monkeypatch.setattr(module, "normalize_columns", lambda frame, cleaning: frame)
    monkeypatch.setattr(module, "enforce_schema", lambda frame, schema, drop_missing_ints: frame)
    monkeypatch.setattr(module, "clean_data", lambda frame, invariants: frame)
    monkeypatch.setattr(module, "validate_min_rows", lambda frame, min_rows: None)

    def _fake_save_data(frame: pd.DataFrame, *, config: Any, data_dir: Path) -> Path:
        captured["saved_rows"] = len(frame)
        captured["data_dir"] = data_dir
        return data_dir / "data.parquet"

    monkeypatch.setattr(module, "save_data", _fake_save_data)
    monkeypatch.setattr(module, "get_memory_usage", lambda frame: 123.4)
    monkeypatch.setattr(
        module,
        "compute_memory_change",
        lambda *, target_metadata, new_memory_usage, stage: {"delta_mb": 10.0, "stage": stage},
    )

    metadata_obj = SimpleNamespace(model_dump=lambda **kwargs: {"ok": True})

    def _fake_prepare_metadata(
        frame: pd.DataFrame,
        *,
        config: Any,
        start_time: float,
        data_path: Path,
        source_data_path: Path,
        source_data_format: str,
        owner: str,
        memory_info: dict[str, Any],
        interim_run_id: str,
    ) -> Any:
        captured["prepare"] = {
            "rows": len(frame),
            "data_path": data_path,
            "source_data_path": source_data_path,
            "source_data_format": source_data_format,
            "owner": owner,
            "memory_info": memory_info,
            "interim_run_id": interim_run_id,
        }
        return metadata_obj

    monkeypatch.setattr(module, "prepare_metadata", _fake_prepare_metadata)
    monkeypatch.setattr(
        module,
        "save_metadata",
        lambda payload, target_dir: captured.setdefault("metadata", (payload, target_dir)),
    )

    code = module.main()

    expected_data_dir = Path("data/interim/hotel_bookings/v1/20260306T150000_abcdef01")
    actual_data_dir = Path(captured["data_dir"])
    actual_log_path = Path(captured["log"][0])

    assert code == 0
    assert captured["log"][1] == module.logging.INFO
    assert actual_log_path.name == "build_interim_dataset.log"
    assert actual_log_path.parent.parts[-5:] == expected_data_dir.parts
    assert captured["saved_rows"] == 1
    assert actual_data_dir.parts[-5:] == expected_data_dir.parts
    assert captured["prepare"]["owner"] == "Sebastijan"
    assert captured["prepare"]["source_data_format"] == "csv"
    assert captured["prepare"]["interim_run_id"] == "20260306T150000_abcdef01"
    assert captured["metadata"][0] == {"ok": True}
    assert Path(captured["metadata"][1]).parts[-5:] == expected_data_dir.parts


def test_main_returns_resolved_exit_code_when_config_load_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Map early configuration-loading failures through the common exit resolver."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            data="hotel_bookings",
            version="v1",
            raw_snapshot_id="latest",
            logging_level="INFO",
            owner="Sebastijan",
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T150500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    err = FileNotFoundError("missing config")

    def _raise(_path: Path) -> dict[str, Any]:
        raise err

    monkeypatch.setattr(module, "load_yaml", _raise)
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 71 if e is err else 99)

    code = module.main()

    assert code == 71


def test_main_returns_resolved_exit_code_when_validate_min_rows_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return mapped exit code for domain validation failures during processing."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            data="hotel_bookings",
            version="v1",
            raw_snapshot_id="latest",
            logging_level="INFO",
            owner="Sebastijan",
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T151000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="8899aabbccddeeff"))
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    raw_snapshot_path = tmp_path / "data" / "raw" / "hotel_bookings" / "v_raw" / "run_010"
    raw_snapshot_path.mkdir(parents=True)

    config = SimpleNamespace(
        raw_data_version="v_raw",
        cleaning=SimpleNamespace(),
        data_schema=SimpleNamespace(),
        drop_missing_ints=False,
        invariants=SimpleNamespace(),
        min_rows=10,
        drop_duplicates=False,
    )

    monkeypatch.setattr(module, "load_yaml", lambda _path: {"cfg": 1})
    monkeypatch.setattr(module, "validate_config", lambda raw, type: config)
    monkeypatch.setattr(module, "get_snapshot_path", lambda _sid, _parent: raw_snapshot_path)
    monkeypatch.setattr(module, "load_json", lambda _path: {"data": {"suffix": "data.csv"}})
    monkeypatch.setattr(module, "get_data_suffix_and_format", lambda meta, location: ("data.csv", "csv"))
    monkeypatch.setattr(module, "validate_data", lambda data_path, metadata: None)
    monkeypatch.setattr(module, "read_data", lambda fmt, path: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(module, "normalize_columns", lambda frame, cleaning: frame)
    monkeypatch.setattr(module, "enforce_schema", lambda frame, schema, drop_missing_ints: frame)
    monkeypatch.setattr(module, "clean_data", lambda frame, invariants: frame)

    err = ValueError("not enough rows")

    def _raise_min_rows(frame: pd.DataFrame, min_rows: int) -> None:
        raise err

    monkeypatch.setattr(module, "validate_min_rows", _raise_min_rows)
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 64 if e is err else 99)

    code = module.main()

    assert code == 64
