"""Integration tests for `pipelines.features.freeze` CLI flow.

Tests stub out registry loading and strategy execution to verify the
high-level orchestration and metadata persistence.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pipelines.features.freeze as freeze_mod


def test_freeze_feature_set_success(tmp_path: Path, monkeypatch: Any) -> None:
    args = SimpleNamespace(feature_set="fs", version="v", snapshot_binding_key=None, owner="me", logging_level="INFO")
    monkeypatch.setattr(freeze_mod, "parse_args", lambda: args)
    monkeypatch.setattr(freeze_mod, "bootstrap_logging", lambda *a, **k: None)
    monkeypatch.setattr(freeze_mod, "add_file_handler", lambda *a, **k: None)

    # load_feature_registry -> raw config; validate_feature_registry -> normalized config object
    monkeypatch.setattr(freeze_mod, "load_feature_registry", lambda fs, v: {"dummy": True})
    monkeypatch.setattr(freeze_mod, "get_strategy_type", lambda cfg: "tabular")
    monkeypatch.setattr(freeze_mod, "validate_feature_registry", lambda raw, t: SimpleNamespace(type="tabular", feature_store_path=str(tmp_path / "feature_store")))

    # Fake strategy that returns an output with snapshot_path and metadata
    snapshot_dir = tmp_path / "feature_store" / "snap1"

    class FakeStrategy:
        def freeze(self, *a, **k):
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(snapshot_path=snapshot_dir, metadata={"ok": True})

    monkeypatch.setattr(freeze_mod, "get_strategy", lambda t: FakeStrategy())

    saved: dict[str, Any] = {}

    def fake_save_metadata(metadata, target_dir: Path):
        saved["meta"] = metadata
        saved["target"] = Path(target_dir)

    monkeypatch.setattr(freeze_mod, "save_metadata", fake_save_metadata)

    rc = freeze_mod.main()
    assert rc == 0
    assert saved["target"] == snapshot_dir
