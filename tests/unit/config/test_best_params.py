"""Unit tests for best-parameter merge utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ml.config.best_params import apply_best_params, unflatten_best_params
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


def test_unflatten_best_params_groups_known_keys_and_preserves_unknowns() -> None:
    """Map known model/ensemble fields while keeping non-mapped keys unchanged."""
    flat = {
        "Model__depth": 8,
        "learning_rate": 0.07,
        "bagging_temperature": 1.5,
        "iterations": 700,
        "search__custom__token": "keep-as-is",
    }

    result = unflatten_best_params(flat)

    assert result == {
        "iterations": 700,
        "search__custom__token": "keep-as-is",
        "model": {"depth": 8, "learning_rate": 0.07},
        "ensemble": {"bagging_temperature": 1.5},
    }


def test_apply_best_params_merges_structured_params_into_selected_target(tmp_path: Path) -> None:
    """Merge persisted best params into configured section without dropping existing keys."""
    metadata_path = tmp_path / "metadata.json"
    payload: dict[str, Any] = {
        "search_results": {
            "best_model_params": {
                "Model__depth": 10,
                "bagging_temperature": 0.8,
                "iterations": 1200,
            }
        }
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = {
        "training": {
            "existing": "value",
            "model": {"learning_rate": 0.1},
        }
    }

    merged = apply_best_params(cfg, metadata_path, merge_target="training", strict=True)

    assert merged["training"]["existing"] == "value"
    assert merged["training"]["iterations"] == 1200
    assert merged["training"]["model"] == {"learning_rate": 0.1, "depth": 10}
    assert merged["training"]["ensemble"] == {"bagging_temperature": 0.8}


def test_apply_best_params_returns_original_config_when_file_missing_and_non_strict(
    tmp_path: Path,
) -> None:
    """Skip merge and return original object when metadata file is absent in non-strict mode."""
    cfg = {"training": {"iterations": 500}}

    result = apply_best_params(
        cfg,
        tmp_path / "missing_metadata.json",
        strict=False,
    )

    assert result is cfg


def test_apply_best_params_raises_when_file_missing_and_strict(tmp_path: Path) -> None:
    """Raise ConfigError for missing metadata file when strict mode is enabled."""
    with pytest.raises(ConfigError, match="best_params file not found"):
        apply_best_params(
            {"training": {"iterations": 500}},
            tmp_path / "missing_metadata.json",
            strict=True,
        )


def test_apply_best_params_raises_when_best_params_are_missing_in_strict_mode(
    tmp_path: Path,
) -> None:
    """Raise `ConfigError` when search metadata does not contain `best_model_params`."""
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"search_results": {}}), encoding="utf-8")

    with pytest.raises(ConfigError, match="No best_params found"):
        apply_best_params(
            {"training": {}},
            metadata_path,
            strict=True,
        )


def test_apply_best_params_returns_original_when_best_params_missing_and_non_strict(
    tmp_path: Path,
) -> None:
    """Return original config when best params are absent and strict mode is disabled."""
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"search_results": {}}), encoding="utf-8")
    cfg = {"training": {"iterations": 500}}

    result = apply_best_params(cfg, metadata_path, strict=False)

    assert result is cfg


def test_apply_best_params_returns_original_on_invalid_json_in_non_strict_mode(
    tmp_path: Path,
) -> None:
    """Fallback to original config when metadata content is invalid and strict is disabled."""
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{not-json", encoding="utf-8")
    cfg = {"training": {"iterations": 500}}

    result = apply_best_params(cfg, metadata_path, strict=False)

    assert result is cfg
