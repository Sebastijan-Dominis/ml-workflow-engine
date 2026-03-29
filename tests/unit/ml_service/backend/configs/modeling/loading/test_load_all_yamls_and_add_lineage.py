"""Unit tests for load_all_yamls_and_add_lineage.

These tests exercise YAML parsing, lineage injection, and error paths.
"""
from __future__ import annotations

from typing import Any

import pytest
from ml_service.backend.configs.modeling.loading.load_all_yamls_and_add_lineage import (
    load_all_yamls_and_add_lineage,
)
from ml_service.backend.configs.modeling.models.configs import RawConfigsWithLineage


def _model_specs_yaml(with_lineage: bool = True) -> str:
    if with_lineage:
        return """
model_specs_lineage: {}
models:
  my_model: {}
"""
    return """
models:
  my_model: {}
"""


def _search_yaml(with_lineage: bool = True) -> str:
    if with_lineage:
        return """
search_lineage: {}
search:
  extends: []
"""
    return """
search:
  extends: []
"""


def _training_yaml(with_lineage: bool = True) -> str:
    if with_lineage:
        return """
training_lineage: {}
training:
  param: value
"""
    return """
training:
  param: value
"""


def test_load_all_yamls_and_add_lineage_success(monkeypatch: Any) -> None:
    # Make timestamps deterministic
    monkeypatch.setattr("ml_service.backend.configs.formatting.timestamp.utc_timestamp", lambda: "2026-03-29T12:00:00Z")

    payload = {
        "model_specs": _model_specs_yaml(True),
        "search": _search_yaml(True),
        "training": _training_yaml(True),
    }

    out = load_all_yamls_and_add_lineage(payload)
    assert isinstance(out, RawConfigsWithLineage)

    assert "model_specs_lineage" in out.model_specs
    assert out.model_specs["model_specs_lineage"]["created_at"] == "2026-03-29T12:00:00Z"

    assert "search_lineage" in out.search
    assert out.search["search_lineage"]["created_at"] == "2026-03-29T12:00:00Z"

    assert "training_lineage" in out.training
    assert out.training["training_lineage"]["created_at"] == "2026-03-29T12:00:00Z"


def test_missing_lineage_key_raises() -> None:
    # model_specs missing lineage key should cause add_timestamp to raise
    payload = {
        "model_specs": _model_specs_yaml(False),
        "search": _search_yaml(True),
        "training": _training_yaml(True),
    }

    with pytest.raises(ValueError) as exc:
        load_all_yamls_and_add_lineage(payload)

    assert "Missing 'model_specs_lineage'" in str(exc.value)


def test_invalid_yaml_raises() -> None:
    payload = {
        "model_specs": "foo: [unclosed",
        "search": _search_yaml(True),
        "training": _training_yaml(True),
    }

    with pytest.raises(ValueError) as exc:
        load_all_yamls_and_add_lineage(payload)

    assert "YAML parsing error" in str(exc.value)


def test_empty_configs_raise() -> None:
    # Empty model_specs string should be treated as empty and trigger the empty-config error
    payload = {
        "model_specs": "",
        "search": _search_yaml(True),
        "training": _training_yaml(True),
    }

    with pytest.raises(ValueError) as exc:
        load_all_yamls_and_add_lineage(payload)

    assert "One or more configs are empty or invalid YAML" in str(exc.value)
