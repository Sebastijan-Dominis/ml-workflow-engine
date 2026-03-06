"""Unit tests for data-configuration hash helper."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from ml.config import compute_data_config_hash as hash_module
from ml.exceptions import RuntimeMLError

pytestmark = pytest.mark.unit


def test_compute_data_config_hash_is_stable_for_equal_payloads() -> None:
    """Produce the same digest for semantically identical mapping payloads."""
    left_payload: dict[str, Any] = {"b": 2, "a": 1, "nested": {"y": 9, "x": 8}}
    right_payload: dict[str, Any] = {"nested": {"x": 8, "y": 9}, "a": 1, "b": 2}

    left_hash = hash_module.compute_data_config_hash(left_payload)  # type: ignore[arg-type]
    right_hash = hash_module.compute_data_config_hash(right_payload)  # type: ignore[arg-type]

    assert left_hash == right_hash
    assert len(left_hash) == 32


def test_compute_data_config_hash_uses_yaml_sort_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serialize config with ``sort_keys=True`` to enforce deterministic hashing."""
    captured: dict[str, Any] = {}

    def _fake_dump(config: Any, sort_keys: bool) -> str:
        captured["config"] = config
        captured["sort_keys"] = sort_keys
        return "ordered-payload"

    monkeypatch.setattr(hash_module.yaml, "dump", _fake_dump)

    payload = {"k": "v"}
    result = hash_module.compute_data_config_hash(payload)  # type: ignore[arg-type]

    assert captured["config"] is payload
    assert captured["sort_keys"] is True
    assert result == hashlib.md5(b"ordered-payload").hexdigest()


def test_compute_data_config_hash_wraps_serialization_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wrap YAML serialization failures as RuntimeMLError and emit error logs."""

    def _failing_dump(config: Any, sort_keys: bool) -> str:
        _ = (config, sort_keys)
        raise ValueError("boom")

    monkeypatch.setattr(hash_module.yaml, "dump", _failing_dump)

    with caplog.at_level("ERROR", logger=hash_module.__name__), pytest.raises(
        RuntimeMLError,
        match="Error computing config hash\\. ",
    ):
        hash_module.compute_data_config_hash({"a": 1})  # type: ignore[arg-type]

    assert "Error computing config hash. Details: boom" in caplog.text
