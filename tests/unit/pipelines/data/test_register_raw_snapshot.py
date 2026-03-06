"""Unit tests for raw snapshot registration CLI behavior."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from pipelines.data import register_raw_snapshot as module

pytestmark = pytest.mark.unit


def test_parse_args_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse default snapshot/logging/owner values when optional flags are omitted."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["register_raw_snapshot", "--data", "hotel_bookings", "--version", "v1"],
    )

    args = module.parse_args()

    assert args.data == "hotel_bookings"
    assert args.version == "v1"
    assert args.snapshot_id == "latest"
    assert args.logging_level == "INFO"
    assert args.owner == "Sebastijan"


def test_main_happy_path_reads_data_and_persists_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Read the single raw data file and save generated metadata successfully."""
    snapshot_dir = tmp_path / "data" / "raw" / "hotel_bookings" / "v1" / "run_001"
    snapshot_dir.mkdir(parents=True)
    data_file = snapshot_dir / "data.csv"
    data_file.write_text("col\n1\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            data="hotel_bookings",
            version="v1",
            snapshot_id="latest",
            logging_level="INFO",
            owner="Sebastijan",
        ),
    )
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: captured.setdefault("bootstrap", level))
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: captured.setdefault("log", (path, level)))
    monkeypatch.setattr(module, "get_snapshot_path", lambda _sid, _parent: snapshot_dir)

    df = pd.DataFrame({"col": [1]})
    monkeypatch.setattr(module, "read_data", lambda fmt, path: (captured.setdefault("read", (fmt, path)), df)[1])

    metadata_payload = {"raw_run_id": "run_001"}
    monkeypatch.setattr(
        module,
        "prepare_metadata",
        lambda frame, *, args, data_path, raw_run_id, data_format, data_suffix: (
            captured.setdefault(
                "meta_inputs",
                {
                    "rows": len(frame),
                    "args": args,
                    "data_path": data_path,
                    "raw_run_id": raw_run_id,
                    "data_format": data_format,
                    "data_suffix": data_suffix,
                },
            ),
            SimpleNamespace(model_dump=lambda **kwargs: metadata_payload),
        )[1],
    )
    monkeypatch.setattr(
        module,
        "save_metadata",
        lambda payload, target_dir: captured.setdefault("save", (payload, target_dir)),
    )

    code = module.main()

    assert code == 0
    assert captured["read"] == ("csv", data_file)
    assert captured["meta_inputs"]["raw_run_id"] == "run_001"
    assert captured["meta_inputs"]["data_format"] == "csv"
    assert captured["meta_inputs"]["data_suffix"] == "data.csv"
    assert captured["save"] == (metadata_payload, snapshot_dir)


def test_main_returns_resolved_code_when_snapshot_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate snapshot path resolution failures via shared exit-code resolver."""
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            data="hotel_bookings",
            version="v1",
            snapshot_id="latest",
            logging_level="INFO",
            owner="Sebastijan",
        ),
    )
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)

    err = RuntimeError("missing snapshot")

    def _raise(_snapshot_id: str, _parent_dir: Path) -> Path:
        raise err

    monkeypatch.setattr(module, "get_snapshot_path", _raise)
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 31 if e is err else 99)

    code = module.main()

    assert code == 31


def test_main_returns_resolved_code_for_invalid_data_file_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return mapped exit code when snapshot does not contain exactly one data file."""
    snapshot_dir = tmp_path / "data" / "raw" / "hotel_bookings" / "v1" / "run_002"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "data.csv").write_text("a\n1\n", encoding="utf-8")
    (snapshot_dir / "data.parquet").write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            data="hotel_bookings",
            version="v1",
            snapshot_id="run_002",
            logging_level="INFO",
            owner="Sebastijan",
        ),
    )
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: None)
    monkeypatch.setattr(module, "get_snapshot_path", lambda _sid, _parent: snapshot_dir)
    monkeypatch.setattr(module, "resolve_exit_code", lambda _e: 41)

    code = module.main()

    assert code == 41


def test_main_returns_resolved_code_when_read_data_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Map downstream read errors through the CLI exit-code resolver."""
    snapshot_dir = tmp_path / "data" / "raw" / "hotel_bookings" / "v1" / "run_003"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "data.csv").write_text("a\n1\n", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            data="hotel_bookings",
            version="v1",
            snapshot_id="run_003",
            logging_level="INFO",
            owner="Sebastijan",
        ),
    )
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: None)
    monkeypatch.setattr(module, "get_snapshot_path", lambda _sid, _parent: snapshot_dir)

    err = ValueError("cannot parse")

    def _raise(_fmt: str, _path: Path) -> pd.DataFrame:
        raise err

    monkeypatch.setattr(module, "read_data", _raise)
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 22 if e is err else 99)

    code = module.main()

    assert code == 22
