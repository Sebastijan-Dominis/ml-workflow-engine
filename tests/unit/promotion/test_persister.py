"""Unit tests for promotion persistence orchestration and decision messaging."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from ml.exceptions import RuntimeMLError
from ml.promotion.config.promotion_thresholds import Direction, MetricName, MetricSet
from ml.promotion.constants.constants import (
    PreviousProductionRunIdentity,
    ProductionComparisonResult,
    ThresholdComparisonResult,
)
from ml.promotion.context import PromotionContext
from ml.promotion.persister import PromotionPersister
from ml.promotion.result import PromotionResult
from ml.promotion.state import PromotionState

pytestmark = pytest.mark.unit


def _context(*, stage: str) -> PromotionContext:
    """Build minimal promotion context used by PromotionPersister methods."""
    args = SimpleNamespace(stage=stage, problem="cancellation", segment="city_hotel")
    paths = SimpleNamespace(
        run_dir=Path("model_registry") / "runs" / "run-1",
        registry_path=Path("model_registry") / "models.yaml",
        archive_path=Path("model_registry") / "archive.yaml",
        train_run_dir=Path("experiments") / "training" / "train-1",
    )
    return cast(PromotionContext, SimpleNamespace(args=args, paths=paths, run_id="run-1", timestamp="20260305T000000"))


def _state(*, current_prod_model_info: dict | None = None) -> PromotionState:
    """Build minimal promotion state with configurable current production model payload."""
    return cast(
        PromotionState,
        SimpleNamespace(
            model_registry={"registry": 1},
            archive_registry={"archive": 1},
            evaluation_metrics={"val": {"f1": 0.8}},
            promotion_thresholds=SimpleNamespace(model_dump=lambda: {"thresholds": 1}),
            current_prod_model_info=current_prod_model_info,
            previous_production_run_identity=PreviousProductionRunIdentity(
                experiment_id=None,
                train_run_id=None,
                eval_run_id=None,
                explain_run_id=None,
                promotion_id=None,
            ),
            git_commit="commit-1",
            threshold_comparison=ThresholdComparisonResult(
                meets_thresholds=True,
                message="threshold ok",
                target_sets=[MetricSet.VAL],
                target_metrics=[MetricName.F1],
                directions={MetricName.F1: Direction.MAXIMIZE},
            ),
        ),
    )


def test_build_reason_and_log_msg_for_successful_production_promotion() -> None:
    """Return production success reason and archive log message with previous promotion id context."""
    persister = PromotionPersister()
    context = _context(stage="production")
    state = _state(current_prod_model_info={"promotion_id": "prev-prom-42"})
    result = PromotionResult(promotion_decision=True, beats_previous=True, previous_production_metrics=None, run_info={"x": 1})

    reason, log_msg = persister._build_reason_and_log_msg(context, state, result)

    assert reason == "Model meets all promotion criteria."
    assert "prev-prom-42" in log_msg


def test_build_reason_and_log_msg_for_failed_production_combines_reasons() -> None:
    """Combine threshold and production-comparison failures into a single reason string."""
    persister = PromotionPersister()
    context = _context(stage="production")
    state = _state(current_prod_model_info={"promotion_id": "prev-prom-42"})
    state.threshold_comparison = ThresholdComparisonResult(
        meets_thresholds=False,
        message="threshold miss",
        target_sets=[MetricSet.VAL],
        target_metrics=[MetricName.F1],
        directions={MetricName.F1: Direction.MAXIMIZE},
    )
    result = PromotionResult(
        promotion_decision=False,
        beats_previous=False,
        previous_production_metrics={"val": {"f1": 0.79}},
        production_comparison=ProductionComparisonResult(
            beats_previous=False,
            message="production miss",
            previous_production_metrics={"val": {"f1": 0.79}},
        ),
    )

    reason, log_msg = persister._build_reason_and_log_msg(context, state, result)

    assert reason == "threshold miss; production miss"
    assert "Model promotion criteria not met" in log_msg


def test_build_reason_and_log_msg_for_successful_staging_promotion() -> None:
    """Return staging-specific success reason and concise staging log message."""
    persister = PromotionPersister()
    context = _context(stage="staging")
    state = _state()
    result = PromotionResult(promotion_decision=True, beats_previous=True, previous_production_metrics=None, run_info={"x": 1})

    reason, log_msg = persister._build_reason_and_log_msg(context, state, result)

    assert reason == "Model beats the thresholds. No comparison against production model for staging promotion."
    assert log_msg == "Model promoted to staging successfully."


def test_build_reason_and_log_msg_for_failed_staging_uses_threshold_message() -> None:
    """Return threshold-comparison message directly when staging promotion criteria are not met."""
    persister = PromotionPersister()
    context = _context(stage="staging")
    state = _state()
    state.threshold_comparison = ThresholdComparisonResult(
        meets_thresholds=False,
        message="staging threshold miss",
        target_sets=[MetricSet.VAL],
        target_metrics=[MetricName.F1],
        directions={MetricName.F1: Direction.MAXIMIZE},
    )
    result = PromotionResult(promotion_decision=False, beats_previous=False, previous_production_metrics=None)

    reason, log_msg = persister._build_reason_and_log_msg(context, state, result)

    assert reason == "staging threshold miss"
    assert log_msg == "Model staging criteria not met. Reasoning: staging threshold miss"


def test_build_reason_and_log_msg_raises_when_failed_production_has_no_comparison() -> None:
    """Raise RuntimeMLError when production decision is negative but comparison payload is missing."""
    persister = PromotionPersister()
    context = _context(stage="production")
    state = _state(current_prod_model_info={"promotion_id": "prev-prom-42"})
    result = PromotionResult(promotion_decision=False, beats_previous=False, previous_production_metrics=None)

    with pytest.raises(RuntimeMLError, match="Production comparison result is missing"):
        persister._build_reason_and_log_msg(context, state, result)


def test_build_reason_and_log_msg_raises_when_stage_is_invalid_for_positive_decision() -> None:
    """Raise RuntimeMLError for unsupported stage values even when decision is positive."""
    persister = PromotionPersister()
    context = _context(stage="shadow")
    state = _state()
    result = PromotionResult(promotion_decision=True, beats_previous=True, previous_production_metrics=None, run_info={"x": 1})

    with pytest.raises(RuntimeMLError, match="Invalid stage 'shadow'"):
        persister._build_reason_and_log_msg(context, state, result)


def test_build_reason_and_log_msg_raises_when_stage_is_invalid_for_negative_decision() -> None:
    """Raise RuntimeMLError for unsupported stage values when decision is negative."""
    persister = PromotionPersister()
    context = _context(stage="shadow")
    state = _state()
    result = PromotionResult(promotion_decision=False, beats_previous=False, previous_production_metrics=None)

    with pytest.raises(RuntimeMLError, match="Invalid stage 'shadow'"):
        persister._build_reason_and_log_msg(context, state, result)


def test_persist_raises_when_positive_decision_has_no_run_info() -> None:
    """Raise RuntimeMLError when promotion strategy returns positive decision without run_info."""
    persister = PromotionPersister()
    context = _context(stage="staging")
    state = _state()
    result = PromotionResult(promotion_decision=True, beats_previous=True, previous_production_metrics=None, run_info=None)

    with pytest.raises(RuntimeMLError, match="run_info is missing"):
        persister.persist(context, state, result)


def test_persist_without_promotion_skips_registry_updates_and_saves_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip registry mutations when decision is negative, while still persisting promotion metadata."""
    persister = PromotionPersister()
    context = _context(stage="staging")
    state = _state()
    state.threshold_comparison = ThresholdComparisonResult(
        meets_thresholds=False,
        message="threshold miss",
        target_sets=[MetricSet.VAL],
        target_metrics=[MetricName.F1],
        directions={MetricName.F1: Direction.MAXIMIZE},
    )
    result = PromotionResult(promotion_decision=False, beats_previous=False, previous_production_metrics={"val": {"f1": 0.75}})

    calls: list[str] = []

    def _update_registry_and_archive(**kwargs) -> dict:
        calls.append("update_registry")
        return {"updated": 1}

    def _persist_registry_diff(**kwargs) -> None:
        calls.append("persist_diff")

    def _prepare_metadata(**kwargs) -> dict:
        return {"metadata": 1}

    def _save_metadata(**kwargs) -> None:
        calls.append(f"save_metadata:{kwargs['target_dir'] == context.paths.run_dir}")

    monkeypatch.setattr("ml.promotion.persister.update_registry_and_archive", _update_registry_and_archive)
    monkeypatch.setattr("ml.promotion.persister.persist_registry_diff", _persist_registry_diff)
    monkeypatch.setattr("ml.promotion.persister.prepare_metadata", _prepare_metadata)
    monkeypatch.setattr("ml.promotion.persister.save_metadata", _save_metadata)

    persister.persist(context, state, result)

    assert calls == ["save_metadata:True"]


def test_persist_with_promotion_updates_registry_persists_diff_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Update registry, persist registry diff, and save metadata when decision is positive."""
    persister = PromotionPersister()
    context = _context(stage="production")
    state = _state(current_prod_model_info={"promotion_id": "prev-prom-42"})
    result = PromotionResult(promotion_decision=True, beats_previous=True, previous_production_metrics=None, run_info={"run": 1})

    calls: list[str] = []

    def _update_registry_and_archive_2(**kwargs) -> dict:
        calls.append("update_registry")
        return {"updated": 1}

    def _persist_registry_diff_2(**kwargs) -> None:
        calls.append("persist_diff")

    def _prepare_metadata_2(**kwargs) -> dict:
        return {"metadata": 1}

    def _save_metadata_2(**kwargs) -> None:
        calls.append("save_metadata")

    monkeypatch.setattr("ml.promotion.persister.update_registry_and_archive", _update_registry_and_archive_2)
    monkeypatch.setattr("ml.promotion.persister.persist_registry_diff", _persist_registry_diff_2)
    monkeypatch.setattr("ml.promotion.persister.prepare_metadata", _prepare_metadata_2)
    monkeypatch.setattr("ml.promotion.persister.save_metadata", _save_metadata_2)

    persister.persist(context, state, result)

    assert calls == ["update_registry", "persist_diff", "save_metadata"]
