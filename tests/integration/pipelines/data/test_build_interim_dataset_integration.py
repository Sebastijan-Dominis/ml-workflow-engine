from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from pathlib import Path

import pandas as pd
import pytest

import pipelines.data.build_interim_dataset as mod

pytestmark = pytest.mark.integration


def test_build_interim_dataset_main_happy_path(tmp_path: Path, monkeypatch: Any) -> None:
    # Run inside tmp_path so created data/interim stays in tempdir
    monkeypatch.chdir(tmp_path)

    args = SimpleNamespace(
        data="hotel_bookings",
        version="v1",
        raw_snapshot_id="latest",
        logging_level="INFO",
        owner="tester",
    )

    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(mod, "load_yaml", lambda p: {})

    config = SimpleNamespace(raw_data_version="raw_v1", cleaning={}, data_schema={}, drop_missing_ints=False, drop_duplicates=False, min_rows=1, invariants=[])
    monkeypatch.setattr(mod, "validate_config", lambda raw, type=None: config)

    # Create a fake raw snapshot with metadata
    snapshot_dir = tmp_path / "data" / "raw" / "hotel_bookings" / "raw_v1" / "snap1"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "metadata.json").write_text("{}")

    monkeypatch.setattr(mod, "get_snapshot_path", lambda sid, parent: snapshot_dir)
    monkeypatch.setattr(mod, "load_json", lambda p: {"meta": True})
    monkeypatch.setattr(mod, "get_data_suffix_and_format", lambda metadata, location=None: ("data.csv", "csv"))
    monkeypatch.setattr(mod, "validate_data", lambda *a, **k: None)
    monkeypatch.setattr(mod, "read_data", lambda fmt, path: pd.DataFrame({"a": [1, 2, 2]}))

    monkeypatch.setattr(mod, "normalize_columns", lambda df, cleaning: df)
    monkeypatch.setattr(mod, "enforce_schema", lambda df, schema, drop_missing_ints: df)
    monkeypatch.setattr(mod, "clean_data", lambda df, invariants: df)
    monkeypatch.setattr(mod, "validate_min_rows", lambda df, min_rows: None)

    monkeypatch.setattr(mod, "save_data", lambda df, config, data_dir: data_dir / "data.csv")
    monkeypatch.setattr(mod, "get_memory_usage", lambda df: {"mem": 1})
    monkeypatch.setattr(mod, "compute_memory_change", lambda **k: {"delta": 0})
    monkeypatch.setattr(mod, "prepare_metadata", lambda *a, **k: SimpleNamespace(model_dump=lambda exclude_none=True: {"meta": "ok"}))

    called: dict[str, Any] = {}

    def fake_save_metadata(md, target_dir):
        called["md"] = md
        called["target"] = target_dir

    monkeypatch.setattr(mod, "save_metadata", fake_save_metadata)

    rc = mod.main()

    assert rc == 0
    assert "md" in called and "target" in called
