from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import ml.features.loading.features_and_target as mod
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def test_load_features_and_target_basic_flow(tmp_path: Path, monkeypatch: Any) -> None:
    # Minimal model config
    model_cfg = SimpleNamespace(
        feature_store=SimpleNamespace(path=str(tmp_path / "feature_store"), feature_sets=[]),
        target=SimpleNamespace(name="target", version="v1"),
        segmentation=SimpleNamespace(),
        min_rows=1,
    )

    # Fake feature set spec
    fs_spec = SimpleNamespace(name="fs1", version="v1", file_name="fs1.csv", data_format="csv")

    snapshot_path = tmp_path / "snap1"
    snapshot_path.mkdir()

    # Metadata with required fields
    metadata = {
        "feature_schema_hash": "fs_hash",
        "operator_hash": "op_hash",
        "feature_type": "tabular",
        "data_lineage": [
            {
                "ref": "r",
                "name": "n",
                "version": "v",
                "format": "csv",
                "path_suffix": "p",
                "merge_key": ("id",),
                "merge_how": None,
                "merge_validate": None,
                "snapshot_id": "s",
                "path": "p",
                "loader_validation_hash": "h",
                "data_hash": "h",
                "row_count": 2,
                "column_count": 2,
            }
        ],
        "in_memory_hash": "m",
        "file_hash": "f",
        "entity_key": "id",
    }

    sel = {"fs_spec": fs_spec, "snapshot_path": snapshot_path, "metadata": metadata}

    # Monkeypatch internal helpers to avoid heavy IO/validation
    monkeypatch.setattr(mod, "DataLineageEntry", lambda **kwargs: tuple(sorted(kwargs.items())))
    monkeypatch.setattr(mod, "read_data", lambda fmt, p: pd.DataFrame({"id": [1, 2], "f1": [10, 20]}))
    monkeypatch.setattr(mod, "validate_entity_key", lambda df, key: None)
    monkeypatch.setattr(mod, "validate_feature_set", lambda *a, **k: None)
    monkeypatch.setattr(mod, "load_and_validate_data", lambda dl: None)
    monkeypatch.setattr(mod, "get_target_with_entity_key", lambda data, key, entity_key: pd.DataFrame({"id": [1, 2], "target": [0, 1]}))
    monkeypatch.setattr(mod, "apply_segmentation", lambda data, seg_cfg: data)
    monkeypatch.setattr(mod, "validate_feature_target_entity_key", lambda *a, **k: None)
    monkeypatch.setattr(mod, "validate_min_rows", lambda *a, **k: None)
    monkeypatch.setattr(mod, "validate_target", lambda *a, **k: None)
    monkeypatch.setattr(mod, "validate_and_construct_feature_lineage", lambda raw: [])

    X, y, lineage, entity_key = mod.load_features_and_target(
        cast(mod.SearchModelConfig, model_cfg),
        snapshot_selection=[sel],
        snapshot_binding_key=None,
        drop_entity_key=True,
        strict=True,
    )

    assert entity_key == "id"
    assert "id" not in X.columns
    assert list(y) == [0, 1]
    assert isinstance(lineage, list)
