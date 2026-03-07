"""Unit tests for model-configuration hashing helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.config import hashing as module
from ml.config.schemas.model_cfg import SearchModelConfig
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


def test_compute_model_config_hash_is_order_stable_and_ignores_meta() -> None:
    """Compute identical hashes for equivalent payloads regardless of key order and ``_meta`` content."""
    left = {"b": 2, "a": 1, "nested": {"x": 1, "y": 2}, "_meta": {"foo": "bar"}}
    right = {"nested": {"y": 2, "x": 1}, "a": 1, "b": 2, "_meta": {"baz": "qux"}}

    left_hash = module.compute_model_config_hash(left)
    right_hash = module.compute_model_config_hash(right)

    assert left_hash == right_hash
    assert len(left_hash) == 64


def test_compute_model_config_hash_raises_for_non_dictionary_input() -> None:
    """Raise ``ConfigError`` when hashing input is not a dictionary payload."""
    with pytest.raises(ConfigError, match="Config must be a dictionary"):
        module.compute_model_config_hash(cast(dict[str, Any], ["not", "a", "dict"]))


def test_add_config_hash_updates_meta_config_hash_on_config_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attach computed config hash to ``cfg.meta.config_hash`` and return same object."""
    cfg = cast(
        SearchModelConfig,
        SimpleNamespace(
            meta=SimpleNamespace(config_hash=None),
            model_dump=lambda **_kwargs: {"a": 1, "_meta": {"ignore": True}},
        ),
    )

    monkeypatch.setattr(module, "compute_model_config_hash", lambda payload: "hash-123")

    result = module.add_config_hash(cfg)

    assert result is cfg
    assert cfg.meta.config_hash == "hash-123"
