"""Unit tests for tree-model adapter factory helper."""

from __future__ import annotations

import pytest
from ml.exceptions import PipelineContractError
from ml.runners.explainability.explainers.tree_model.utils.adapter import (
    get_adapter as adapter_module,
)

pytestmark = pytest.mark.unit


def test_get_tree_model_adapter_returns_catboost_adapter_for_supported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return CatBoost adapter when model is instance of configured CatBoost type."""

    class _DummyCatBoost:
        pass

    model = _DummyCatBoost()
    created_with: list[object] = []

    class _DummyAdapter:
        def __init__(self, wrapped_model: object) -> None:
            created_with.append(wrapped_model)

    monkeypatch.setattr(adapter_module, "CatBoost", _DummyCatBoost)
    monkeypatch.setattr(adapter_module, "CatBoostAdapter", _DummyAdapter)

    adapter = adapter_module.get_tree_model_adapter(model)

    assert isinstance(adapter, _DummyAdapter)
    assert created_with == [model]


def test_get_tree_model_adapter_raises_pipeline_contract_error_for_unknown_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Raise PipelineContractError with model-type context when adapter is unavailable."""
    model = object()

    with caplog.at_level("ERROR", logger=adapter_module.__name__), pytest.raises(
        PipelineContractError,
        match="No adapter found for model type: object",
    ):
        adapter_module.get_tree_model_adapter(model)

    assert "No adapter found for model type: object" in caplog.text
