"""Unit tests for explainer factory resolution helper."""

from __future__ import annotations

import pytest
from ml.exceptions import PipelineContractError
from ml.runners.explainability.utils import get_explainer as get_explainer_module

pytestmark = pytest.mark.unit


class _DummyExplainer:
    """Minimal explainer stub used for registry-resolution tests."""


def test_get_explainer_instantiates_registered_explainer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiate and return explainer class mapped to the provided key."""
    monkeypatch.setitem(get_explainer_module.EXPLAINERS, "dummy", _DummyExplainer)

    explainer = get_explainer_module.get_explainer("dummy")

    assert isinstance(explainer, _DummyExplainer)


def test_get_explainer_raises_pipeline_contract_error_for_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise contract error with stable message for unknown explainer keys."""
    monkeypatch.delitem(get_explainer_module.EXPLAINERS, "missing", raising=False)

    with pytest.raises(PipelineContractError, match="No explainer found for algorithm 'missing'."):
        get_explainer_module.get_explainer("missing")


def test_get_explainer_logs_error_for_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit an error log that includes the unresolved explainer key."""
    monkeypatch.delitem(get_explainer_module.EXPLAINERS, "unknown", raising=False)

    with caplog.at_level("ERROR", logger=get_explainer_module.__name__), pytest.raises(
        PipelineContractError
    ):
        get_explainer_module.get_explainer("unknown")

    assert "No explainer found for algorithm 'unknown'." in caplog.text


def test_get_explainer_logs_selected_class(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit debug log with selected explainer class and registry key."""
    monkeypatch.setitem(get_explainer_module.EXPLAINERS, "dummy", _DummyExplainer)

    with caplog.at_level("DEBUG", logger=get_explainer_module.__name__):
        get_explainer_module.get_explainer("dummy")

    assert "Using explainer _DummyExplainer for algorithm=dummy" in caplog.text


def test_get_explainer_propagates_constructor_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate explainer constructor exceptions without wrapping."""

    class _BrokenExplainer:
        def __init__(self) -> None:
            raise RuntimeError("constructor failed")

    monkeypatch.setitem(get_explainer_module.EXPLAINERS, "broken", _BrokenExplainer)

    with pytest.raises(RuntimeError, match="constructor failed"):
        get_explainer_module.get_explainer("broken")
