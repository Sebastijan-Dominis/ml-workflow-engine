from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ml.pipelines.validation as mod
import pytest
from ml.exceptions import ConfigError

pytestmark = pytest.mark.integration


def _base_pipeline_cfg() -> dict[str, Any]:
    return {
        "name": "p",
        "version": "v1",
        "steps": ["SchemaValidator"],
        "assumptions": {
            "handles_categoricals": True,
            "supports_regression": True,
            "supports_classification": True,
        },
        "lineage": {"created_by": "t", "created_at": "2026-01-01T00:00:00Z"},
    }


def test_validate_pipeline_config_accepts_valid_cfg() -> None:
    cfg = _base_pipeline_cfg()
    validated = mod.validate_pipeline_config(cfg)
    assert validated.name == "p"


def test_validate_pipeline_config_rejects_bad_version() -> None:
    bad = _base_pipeline_cfg()
    bad["version"] = "1"
    with pytest.raises(ConfigError):
        mod.validate_pipeline_config(bad)


def test_validate_pipeline_config_consistency_happy_path(tmp_path: Path, monkeypatch: Any) -> None:
    # Fake load_json to return minimal metadata and validate_search_record to return expected object
    monkeypatch.setattr(mod, "load_json", lambda p: {"metadata": {"pipeline_hash": "h1"}})
    monkeypatch.setattr(mod, "validate_search_record", lambda raw: SimpleNamespace(metadata=SimpleNamespace(pipeline_hash="h1")))

    # Should not raise
    mod.validate_pipeline_config_consistency(actual_hash="h1", search_dir=tmp_path)


def test_validate_pipeline_config_consistency_mismatch_raises(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(mod, "load_json", lambda p: {"metadata": {"pipeline_hash": "h_expected"}})
    monkeypatch.setattr(mod, "validate_search_record", lambda raw: SimpleNamespace(metadata=SimpleNamespace(pipeline_hash="h_expected")))

    with pytest.raises(ConfigError):
        mod.validate_pipeline_config_consistency(actual_hash="h_actual", search_dir=tmp_path)
