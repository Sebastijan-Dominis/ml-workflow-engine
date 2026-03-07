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


def test_compute_model_config_hash_does_not_mutate_original_input() -> None:
    """Leave caller payload unchanged while removing ``_meta`` only from internal copy."""
    cfg = {"alpha": 1, "nested": {"beta": 2}, "_meta": {"note": "keep-me"}}

    _ = module.compute_model_config_hash(cfg)

    assert cfg == {"alpha": 1, "nested": {"beta": 2}, "_meta": {"note": "keep-me"}}


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


def test_add_config_hash_calls_model_dump_with_expected_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request alias-preserving dump with top-level ``_meta`` excluded before hashing."""
    captured_dump_kwargs: dict[str, Any] = {}
    captured_payload: dict[str, Any] = {}

    def _model_dump(**kwargs: Any) -> dict[str, Any]:
        captured_dump_kwargs.update(kwargs)
        return {"task": {"type": "classification"}, "_meta": {"ignored": True}}

    def _compute_hash(payload: dict[str, Any]) -> str:
        captured_payload.update(payload)
        return "hash-999"

    cfg = cast(
        SearchModelConfig,
        SimpleNamespace(
            meta=SimpleNamespace(config_hash=None),
            model_dump=_model_dump,
        ),
    )

    monkeypatch.setattr(module, "compute_model_config_hash", _compute_hash)

    module.add_config_hash(cfg)

    assert captured_dump_kwargs == {"exclude": {"_meta"}, "by_alias": True}
    assert captured_payload == {"task": {"type": "classification"}, "_meta": {"ignored": True}}
    assert cfg.meta.config_hash == "hash-999"
