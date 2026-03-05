"""Unit tests for promotion service orchestration."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.exceptions import UserError
from ml.promotion.context import PromotionContext
from ml.promotion.service import PromotionService

pytestmark = pytest.mark.unit


def _context(*, stage: str = "production") -> PromotionContext:
    """Build minimal promotion context payload required by PromotionService."""
    args = SimpleNamespace(
        stage=stage,
        train_run_id="train-1",
        eval_run_id="eval-1",
        explain_run_id="explain-1",
    )
    paths = SimpleNamespace(
        train_run_dir=Path("experiments") / "train-1",
        eval_run_dir=Path("experiments") / "eval-1",
        explain_run_dir=Path("experiments") / "explain-1",
        registry_path=Path("model_registry") / "models.yaml",
    )
    return cast(PromotionContext, SimpleNamespace(args=args, paths=paths, runners_metadata=None))


def test_get_strategy_returns_stage_specific_strategy_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve production and staging strategies through stage-specific factory logic."""
    service = PromotionService()

    class _Prod:
        pass

    class _Stage:
        pass

    monkeypatch.setattr("ml.promotion.service.ProductionPromotionStrategy", _Prod)
    monkeypatch.setattr("ml.promotion.service.StagingPromotionStrategy", _Stage)

    assert isinstance(service._get_strategy("production"), _Prod)
    assert isinstance(service._get_strategy("staging"), _Stage)


def test_get_strategy_raises_for_unknown_stage() -> None:
    """Raise UserError when stage is outside supported production/staging values."""
    service = PromotionService()

    with pytest.raises(UserError, match="Unknown stage specified"):
        service._get_strategy("canary")


def test_validate_enriches_context_with_runners_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call validation dependencies and attach loaded runners metadata to context."""
    service = PromotionService()
    context = _context()
    calls: list[str] = []
    metadata_obj = cast(Any, SimpleNamespace(name="runners-metadata"))

    def _validate_run_dirs(*_args: Any, **_kwargs: Any) -> None:
        calls.append("validate_run_dirs")

    def _get_runners_metadata(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("get_runners_metadata")
        return metadata_obj

    def _validate_run_ids(*_args: Any, **_kwargs: Any) -> None:
        calls.append("validate_run_ids")

    def _validate_explainability(*_args: Any, **_kwargs: Any) -> None:
        calls.append("validate_explainability")

    monkeypatch.setattr("ml.promotion.service.validate_run_dirs", _validate_run_dirs)
    monkeypatch.setattr("ml.promotion.service.get_runners_metadata", _get_runners_metadata)
    monkeypatch.setattr("ml.promotion.service.validate_run_ids", _validate_run_ids)
    monkeypatch.setattr("ml.promotion.service.validate_explainability_artifacts_consistency", _validate_explainability)

    result = service._validate(context)

    assert result is context
    assert context.runners_metadata is metadata_obj
    assert calls == [
        "validate_run_dirs",
        "get_runners_metadata",
        "validate_run_ids",
        "validate_explainability",
    ]


def test_run_executes_loader_strategy_and_persister_inside_registry_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute full run flow in lock scope and persist strategy result with loaded state."""
    service = PromotionService()
    context = _context(stage="production")

    state_obj = cast(Any, SimpleNamespace(name="state"))
    result_obj = cast(Any, SimpleNamespace(name="result"))
    events: list[str] = []

    class _FakeStrategy:
        def execute(self, got_context: PromotionContext, got_state: Any) -> Any:
            assert got_context is context
            assert got_state is state_obj
            events.append("strategy.execute")
            return result_obj

    @contextmanager
    def _fake_registry_lock(got_context: PromotionContext):
        assert got_context is context
        events.append("lock.enter")
        try:
            yield
        finally:
            events.append("lock.exit")

    monkeypatch.setattr(service, "_validate", lambda got_context: got_context)
    monkeypatch.setattr(service, "_registry_lock", _fake_registry_lock)
    monkeypatch.setattr(service._state_loader, "load", lambda got_context: state_obj)
    monkeypatch.setattr(service, "_get_strategy", lambda stage: _FakeStrategy())
    monkeypatch.setattr(
        service._persister,
        "persist",
        lambda got_context, got_state, got_result: events.append(
            f"persist:{got_context is context}:{got_state is state_obj}:{got_result is result_obj}"
        ),
    )

    result = service.run(context)

    assert result is result_obj
    assert events == [
        "lock.enter",
        "strategy.execute",
        "persist:True:True:True",
        "lock.exit",
    ]
