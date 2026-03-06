"""Unit tests for processed-dataset build CLI behavior."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from pipelines.data import build_processed_dataset as module

pytestmark = pytest.mark.unit


def test_parse_args_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse default interim snapshot, logging level, and owner values."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["build_processed_dataset", "--data", "hotel_bookings", "--version", "v1"],
    )

    args = module.parse_args()

    assert args.data == "hotel_bookings"
    assert args.version == "v1"
    assert args.interim_snapshot_id == "latest"
    assert args.logging_level == "INFO"
    assert args.owner == "Sebastijan"


def test_main_happy_path_without_row_id_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run processed pipeline and persist outputs when row-id policy is not required."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            data="hotel_bookings",
            version="v1",
            interim_snapshot_id="latest",
            logging_level="INFO",
            owner="Sebastijan",
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T160000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        module,
        "setup_logging",
        lambda path, level: captured.setdefault("log", (path, level)),
    )

    interim_snapshot_path = tmp_path / "data" / "interim" / "hotel_bookings" / "v_interim" / "run_001"
    interim_snapshot_path.mkdir(parents=True)

    config = SimpleNamespace(
        interim_data_version="v_interim",
        remove_columns=["noise"],
        data=SimpleNamespace(name="hotel_bookings"),
    )

    monkeypatch.setattr(module, "load_yaml", lambda _path: {"cfg": 1})
    monkeypatch.setattr(module, "validate_config", lambda raw, type: config)
    monkeypatch.setattr(module, "get_snapshot_path", lambda _sid, _parent: interim_snapshot_path)
    monkeypatch.setattr(module, "load_json", lambda _path: {"meta": 1})
    monkeypatch.setattr(module, "get_data_suffix_and_format", lambda meta, location: ("data.parquet", "parquet"))
    monkeypatch.setattr(module, "validate_data", lambda data_path, metadata: None)

    df_in = pd.DataFrame({"a": [1], "noise": [9]})
    monkeypatch.setattr(module, "read_data", lambda fmt, path: df_in)
    monkeypatch.setattr(module, "remove_columns", lambda frame, cols: frame.drop(columns=cols))
    monkeypatch.setattr(module, "ROW_ID_REQUIRED", {"different_dataset"})
    monkeypatch.setattr(module, "add_row_id", lambda frame, cfg: (_ for _ in ()).throw(AssertionError("should not be called")))

    def _fake_save_data(frame: pd.DataFrame, *, config: Any, data_dir: Path) -> Path:
        captured["saved_cols"] = list(frame.columns)
        captured["data_dir"] = data_dir
        return data_dir / "data.parquet"

    monkeypatch.setattr(module, "save_data", _fake_save_data)
    monkeypatch.setattr(module, "get_memory_usage", lambda frame: 55.0)
    monkeypatch.setattr(
        module,
        "compute_memory_change",
        lambda *, target_metadata, new_memory_usage, stage: {"stage": stage, "delta": 1.0},
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
        source_data_version: str,
        owner: str,
        memory_info: dict[str, Any],
        processed_run_id: str,
        row_id_info: dict[str, Any] | None,
    ) -> Any:
        captured["prepare"] = {
            "owner": owner,
            "source_data_format": source_data_format,
            "source_data_version": source_data_version,
            "processed_run_id": processed_run_id,
            "row_id_info": row_id_info,
        }
        return metadata_obj

    monkeypatch.setattr(module, "prepare_metadata", _fake_prepare_metadata)
    monkeypatch.setattr(
        module,
        "save_metadata",
        lambda payload, target_dir: captured.setdefault("metadata", (payload, target_dir)),
    )

    code = module.main()

    expected_data_dir = Path("data/processed/hotel_bookings/v1/20260306T160000_abcdef01")
    actual_data_dir = Path(captured["data_dir"])
    actual_log_path = Path(captured["log"][0])

    assert code == 0
    assert captured["log"][1] == module.logging.INFO
    assert actual_log_path.name == "build_processed_dataset.log"
    assert actual_log_path.parent.parts[-5:] == expected_data_dir.parts
    assert actual_data_dir.parts[-5:] == expected_data_dir.parts
    assert captured["saved_cols"] == ["a"]
    assert captured["prepare"]["owner"] == "Sebastijan"
    assert captured["prepare"]["source_data_format"] == "parquet"
    assert captured["prepare"]["source_data_version"] == "v_interim"
    assert captured["prepare"]["processed_run_id"] == "20260306T160000_abcdef01"
    assert captured["prepare"]["row_id_info"] is None
    assert captured["metadata"][0] == {"ok": True}
    assert Path(captured["metadata"][1]).parts[-5:] == expected_data_dir.parts


def test_main_adds_row_id_when_policy_requires_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invoke row-id generation branch and forward row-id metadata when required."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            data="bookings_with_row_id",
            version="v2",
            interim_snapshot_id="latest",
            logging_level="INFO",
            owner="Owner",
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T160500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    interim_snapshot_path = tmp_path / "data" / "interim" / "bookings_with_row_id" / "v_interim" / "run_002"
    interim_snapshot_path.mkdir(parents=True)

    config = SimpleNamespace(
        interim_data_version="v_interim",
        remove_columns=[],
        data=SimpleNamespace(name="bookings_with_row_id"),
    )

    monkeypatch.setattr(module, "load_yaml", lambda _path: {"cfg": 1})
    monkeypatch.setattr(module, "validate_config", lambda raw, type: config)
    monkeypatch.setattr(module, "get_snapshot_path", lambda _sid, _parent: interim_snapshot_path)
    monkeypatch.setattr(module, "load_json", lambda _path: {"meta": 1})
    monkeypatch.setattr(module, "get_data_suffix_and_format", lambda meta, location: ("data.csv", "csv"))
    monkeypatch.setattr(module, "validate_data", lambda data_path, metadata: None)

    monkeypatch.setattr(module, "read_data", lambda fmt, path: pd.DataFrame({"feature": [1, 2]}))
    monkeypatch.setattr(module, "remove_columns", lambda frame, cols: frame)
    monkeypatch.setattr(module, "ROW_ID_REQUIRED", {"bookings_with_row_id"})

    row_id_info = {"row_id_column": "row_id", "seed": 42}
    monkeypatch.setattr(
        module,
        "add_row_id",
        lambda frame, cfg: (frame.assign(row_id=[1, 2]), row_id_info),
    )
    monkeypatch.setattr(module, "save_data", lambda frame, *, config, data_dir: data_dir / "data.csv")
    monkeypatch.setattr(module, "get_memory_usage", lambda frame: 11.1)
    monkeypatch.setattr(module, "compute_memory_change", lambda *, target_metadata, new_memory_usage, stage: {})

    captured: dict[str, Any] = {}
    metadata_obj = SimpleNamespace(model_dump=lambda **kwargs: {"ok": True})

    def _fake_prepare_metadata(
        frame: pd.DataFrame,
        *,
        config: Any,
        start_time: float,
        data_path: Path,
        source_data_path: Path,
        source_data_format: str,
        source_data_version: str,
        owner: str,
        memory_info: dict[str, Any],
        processed_run_id: str,
        row_id_info: dict[str, Any] | None,
    ) -> Any:
        captured["row_id_info"] = row_id_info
        return metadata_obj

    monkeypatch.setattr(module, "prepare_metadata", _fake_prepare_metadata)
    monkeypatch.setattr(module, "save_metadata", lambda payload, target_dir: None)

    code = module.main()

    assert code == 0
    assert captured["row_id_info"] == row_id_info


def test_main_returns_resolved_code_when_config_load_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Map configuration-loading failures to standardized CLI exit codes."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            data="hotel_bookings",
            version="v1",
            interim_snapshot_id="latest",
            logging_level="INFO",
            owner="Sebastijan",
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T161000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="8899aabbccddeeff"))
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    err = FileNotFoundError("missing config")
    monkeypatch.setattr(module, "load_yaml", lambda _path: (_ for _ in ()).throw(err))
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 77 if e is err else 99)

    code = module.main()

    assert code == 77


def test_main_returns_resolved_code_when_validate_data_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Map data validation failures to resolver codes before heavy processing begins."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            data="hotel_bookings",
            version="v3",
            interim_snapshot_id="latest",
            logging_level="INFO",
            owner="Sebastijan",
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T161500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="deadbeefcafebabe"))
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    interim_snapshot_path = tmp_path / "data" / "interim" / "hotel_bookings" / "v_interim" / "run_003"
    interim_snapshot_path.mkdir(parents=True)

    config = SimpleNamespace(
        interim_data_version="v_interim",
        remove_columns=[],
        data=SimpleNamespace(name="hotel_bookings"),
    )

    monkeypatch.setattr(module, "load_yaml", lambda _path: {"cfg": 1})
    monkeypatch.setattr(module, "validate_config", lambda raw, type: config)
    monkeypatch.setattr(module, "get_snapshot_path", lambda _sid, _parent: interim_snapshot_path)
    monkeypatch.setattr(module, "load_json", lambda _path: {"data": {"output": {"suffix": "data.csv"}}})
    monkeypatch.setattr(module, "get_data_suffix_and_format", lambda meta, location: ("data.csv", "csv"))

    err = ValueError("bad metadata/data alignment")
    monkeypatch.setattr(module, "validate_data", lambda data_path, metadata: (_ for _ in ()).throw(err))
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 66 if e is err else 99)

    code = module.main()

    assert code == 66
