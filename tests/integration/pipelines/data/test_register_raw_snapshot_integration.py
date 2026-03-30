"""Integration tests for `pipelines.data.register_raw_snapshot`.

These tests create a temporary raw snapshot directory layout and verify the
CLI flow reads data, prepares metadata and persists it via `save_metadata`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pipelines.data.register_raw_snapshot as reg_mod


def test_register_raw_snapshot_success(tmp_path: Path, monkeypatch: Any) -> None:
    # Create a fake snapshot containing a single CSV file
    data_dir = tmp_path / "data" / "raw" / "hotel_bookings" / "v1" / "snap1"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "data.csv"
    csv_path.write_text("a,b\n1,2\n")

    monkeypatch.setattr(reg_mod, "get_snapshot_path", lambda sid, parent: data_dir)
    monkeypatch.setattr(reg_mod, "bootstrap_logging", lambda *a, **k: None)
    monkeypatch.setattr(reg_mod, "add_file_handler", lambda *a, **k: None)

    # Provide CLI args expected by argparse in the module
    monkeypatch.setattr(
        reg_mod,
        "parse_args",
        lambda: argparse.Namespace(
            data="hotel_bookings",
            version="v1",
            snapshot_id="latest",
            logging_level="INFO",
            owner="test",
        ),
    )

    # Make read_data return a DataFrame (we could use the real reader, but stub for speed)
    monkeypatch.setattr(reg_mod, "read_data", lambda fmt, p: pd.read_csv(p))

    called: dict[str, Any] = {}

    class DummyMeta:
        def model_dump(self, exclude_none=True):
            return {"meta": True}

    monkeypatch.setattr(reg_mod, "prepare_metadata", lambda df, **k: DummyMeta())

    def fake_save_metadata(payload, target_dir: Path):
        called["payload"] = payload
        called["target_dir"] = Path(target_dir)

    monkeypatch.setattr(reg_mod, "save_metadata", fake_save_metadata)

    rc = reg_mod.main()
    assert rc == 0
    assert called["target_dir"] == data_dir


def test_register_raw_snapshot_fails_with_multiple_files(tmp_path: Path, monkeypatch: Any) -> None:
    data_dir = tmp_path / "data" / "raw" / "hotel_bookings" / "v1" / "snap2"
    data_dir.mkdir(parents=True)
    (data_dir / "data.csv").write_text("a,b\n1,2\n")
    (data_dir / "data.parquet").write_text("x")

    monkeypatch.setattr(reg_mod, "get_snapshot_path", lambda sid, parent: data_dir)
    monkeypatch.setattr(reg_mod, "bootstrap_logging", lambda *a, **k: None)
    monkeypatch.setattr(reg_mod, "add_file_handler", lambda *a, **k: None)
    monkeypatch.setattr(reg_mod, "read_data", lambda fmt, p: pd.read_csv(p))

    # Provide CLI args expected by argparse in the module
    monkeypatch.setattr(
        reg_mod,
        "parse_args",
        lambda: argparse.Namespace(
            data="hotel_bookings",
            version="v1",
            snapshot_id="latest",
            logging_level="INFO",
            owner="test",
        ),
    )

    rc = reg_mod.main()
    assert rc != 0
