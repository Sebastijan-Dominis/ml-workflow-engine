from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from pathlib import Path

import pandas as pd
import pytest

import pipelines.data.build_processed_dataset as mod

pytestmark = pytest.mark.integration


def test_build_processed_dataset_main_happy_path(tmp_path: Path, monkeypatch: Any) -> None:
    # Run inside tmp_path so created data/processed stays in tempdir
    monkeypatch.chdir(tmp_path)

    args = SimpleNamespace(
        data="hotel_bookings",
        version="v1",
        interim_snapshot_id="latest",
        logging_level="INFO",
        owner="tester",
    )

    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(mod, "load_yaml", lambda p: {})

    config = SimpleNamespace(interim_data_version="int_v1", remove_columns=[], data=SimpleNamespace(name="hotel_bookings"))
    monkeypatch.setattr(mod, "validate_config", lambda raw, type=None: config)

    # Create a fake interim snapshot with metadata
    snapshot_dir = tmp_path / "data" / "interim" / "hotel_bookings" / "int_v1" / "snap1"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "metadata.json").write_text("{}")

    monkeypatch.setattr(mod, "get_snapshot_path", lambda sid, parent: snapshot_dir)
    monkeypatch.setattr(mod, "load_json", lambda p: {"meta": True})
    monkeypatch.setattr(mod, "get_data_suffix_and_format", lambda metadata, location=None: ("data.csv", "csv"))
    monkeypatch.setattr(mod, "validate_data", lambda *a, **k: None)
    monkeypatch.setattr(mod, "read_data", lambda fmt, path: pd.DataFrame({"a": [1, 2]}))
    monkeypatch.setattr(mod, "remove_columns", lambda df, cols: df)

    # Avoid adding row ids during the test - stub the function
    monkeypatch.setattr(mod, "add_row_id", lambda df, cfg: (df, None))

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
