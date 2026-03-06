"""Unit tests for trainer factory resolution helper."""

from __future__ import annotations

import pytest
from ml.exceptions import PipelineContractError
from ml.runners.training.utils import get_trainer as get_trainer_module

pytestmark = pytest.mark.unit


class _DummyTrainer:
    """Minimal trainer stub used for registry-resolution tests."""


def test_get_trainer_instantiates_registered_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiate and return trainer class mapped to the provided key."""
    monkeypatch.setitem(get_trainer_module.TRAINERS, "dummy", _DummyTrainer)

    trainer = get_trainer_module.get_trainer("dummy")

    assert isinstance(trainer, _DummyTrainer)


def test_get_trainer_raises_pipeline_contract_error_for_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise contract error with stable message for unknown trainer keys."""
    monkeypatch.delitem(get_trainer_module.TRAINERS, "missing", raising=False)

    with pytest.raises(PipelineContractError, match="No trainer found for algorithm 'missing'."):
        get_trainer_module.get_trainer("missing")


def test_get_trainer_logs_error_for_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit an error log that includes the unresolved trainer key."""
    monkeypatch.delitem(get_trainer_module.TRAINERS, "unknown", raising=False)

    with caplog.at_level("ERROR", logger=get_trainer_module.__name__), pytest.raises(
        PipelineContractError
    ):
        get_trainer_module.get_trainer("unknown")

    assert "No trainer found for algorithm 'unknown'." in caplog.text


def test_get_trainer_logs_selected_class(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit debug log with selected trainer class and registry key."""
    monkeypatch.setitem(get_trainer_module.TRAINERS, "dummy", _DummyTrainer)

    with caplog.at_level("DEBUG", logger=get_trainer_module.__name__):
        get_trainer_module.get_trainer("dummy")

    assert "Using trainer _DummyTrainer for algorithm=dummy" in caplog.text


def test_get_trainer_propagates_constructor_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate trainer constructor exceptions without wrapping."""

    class _BrokenTrainer:
        def __init__(self) -> None:
            raise RuntimeError("constructor failed")

    monkeypatch.setitem(get_trainer_module.TRAINERS, "broken", _BrokenTrainer)

    with pytest.raises(RuntimeError, match="constructor failed"):
        get_trainer_module.get_trainer("broken")
