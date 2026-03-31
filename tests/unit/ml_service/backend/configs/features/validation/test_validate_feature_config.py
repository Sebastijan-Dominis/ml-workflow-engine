"""Tests for `ml_service.backend.configs.features.validation.validate_feature_config`.

These tests stub the imported `TabularFeaturesConfig` to avoid constructing
the real, heavy pydantic model while exercising the simple dispatch logic.
"""
from __future__ import annotations

from typing import Any

import pytest
from ml_service.backend.configs.features.validation import validate_feature_config as vmod


def test_tabular_delegates_to_tabular_model(monkeypatch: Any) -> None:
    called: dict[str, Any] = {}

    class FakeTabular:
        def __init__(self, **kwargs: Any) -> None:
            called["kwargs"] = kwargs

        def __repr__(self) -> str:  # pragma: no cover - helper
            return "FakeTabular()"

    monkeypatch.setattr(vmod, "TabularFeaturesConfig", FakeTabular)

    payload = {"type": "tabular", "foo": "bar"}
    res = vmod.validate_feature_config(payload)

    assert isinstance(res, FakeTabular)
    assert called["kwargs"] == payload


def test_unsupported_type_raises() -> None:
    with pytest.raises(ValueError) as exc:
        vmod.validate_feature_config({"type": "unknown"})

    assert "Unsupported feature config type" in str(exc.value)
