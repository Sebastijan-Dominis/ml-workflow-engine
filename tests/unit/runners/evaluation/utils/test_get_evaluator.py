"""Unit tests for evaluator factory resolution helper."""

from __future__ import annotations

import pytest
from ml.exceptions import PipelineContractError
from ml.runners.evaluation.utils import get_evaluator as get_evaluator_module

pytestmark = pytest.mark.unit


class _DummyEvaluator:
    """Minimal evaluator stub used for registry-resolution tests."""


def test_get_evaluator_instantiates_registered_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiate and return evaluator class mapped to the provided key."""
    monkeypatch.setitem(get_evaluator_module.EVALUATORS, "dummy", _DummyEvaluator)

    evaluator = get_evaluator_module.get_evaluator("dummy")

    assert isinstance(evaluator, _DummyEvaluator)


def test_get_evaluator_raises_pipeline_contract_error_for_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise contract error with stable message for unknown evaluator keys."""
    monkeypatch.delitem(get_evaluator_module.EVALUATORS, "missing", raising=False)

    with pytest.raises(PipelineContractError, match="No evaluator found for algorithm 'missing'."):
        get_evaluator_module.get_evaluator("missing")


def test_get_evaluator_logs_error_for_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit an error log that includes the unresolved evaluator key."""
    monkeypatch.delitem(get_evaluator_module.EVALUATORS, "unknown", raising=False)

    with caplog.at_level("ERROR", logger=get_evaluator_module.__name__), pytest.raises(
        PipelineContractError
    ):
        get_evaluator_module.get_evaluator("unknown")

    assert "No evaluator found for algorithm 'unknown'." in caplog.text


def test_get_evaluator_logs_selected_class(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit debug log with selected evaluator class and registry key."""
    monkeypatch.setitem(get_evaluator_module.EVALUATORS, "dummy", _DummyEvaluator)

    with caplog.at_level("DEBUG", logger=get_evaluator_module.__name__):
        get_evaluator_module.get_evaluator("dummy")

    assert "Using evaluator _DummyEvaluator for algorithm=dummy" in caplog.text


def test_get_evaluator_propagates_constructor_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate evaluator constructor exceptions without wrapping."""

    class _BrokenEvaluator:
        def __init__(self) -> None:
            raise RuntimeError("constructor failed")

    monkeypatch.setitem(get_evaluator_module.EVALUATORS, "broken", _BrokenEvaluator)

    with pytest.raises(RuntimeError, match="constructor failed"):
        get_evaluator_module.get_evaluator("broken")
