"""Unit tests for staging and production promotion strategies."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.exceptions import RuntimeMLError
from ml.promotion.config.promotion_thresholds import (
    Direction,
    MetricName,
    MetricSet,
    PromotionThresholds,
)
from ml.promotion.constants.constants import (
    PreviousProductionRunIdentity,
    RunnersMetadata,
    ThresholdComparisonResult,
)
from ml.promotion.context import PromotionContext
from ml.promotion.result import PromotionResult
from ml.promotion.state import PromotionState
from ml.promotion.strategies.production import ProductionPromotionStrategy
from ml.promotion.strategies.staging import StagingPromotionStrategy

pytestmark = pytest.mark.unit


def _runners_metadata() -> RunnersMetadata:
    """Build minimally shaped runners metadata object for strategy tests."""
    return RunnersMetadata(
        training_metadata=cast(Any, SimpleNamespace(owner="train-owner")),
        evaluation_metadata=cast(Any, SimpleNamespace(metrics={"val": {"auc": 0.82}})),
        explainability_metadata=cast(Any, SimpleNamespace(features=["f1", "f2"])),
    )


def _context(*, runners_metadata: object) -> PromotionContext:
    """Build minimal strategy context payload used by strategy execute methods."""
    args = SimpleNamespace(
        experiment_id="exp_1",
        train_run_id="train_1",
        eval_run_id="eval_1",
        explain_run_id="explain_1",
    )
    return cast(
        PromotionContext,
        SimpleNamespace(
            args=args,
            run_id="promo_run_1",
            timestamp="20260306T210000",
            runners_metadata=runners_metadata,
        ),
    )


def _state(
    *,
    threshold_comparison: ThresholdComparisonResult,
    evaluation_metrics: dict[str, Any],
    git_commit: str,
    current_prod_model_info: dict[str, Any] | None = None,
) -> PromotionState:
    """Build minimally shaped typed promotion state for strategy execution tests."""
    return PromotionState(
        model_registry={},
        archive_registry={},
        evaluation_metrics=evaluation_metrics,
        promotion_thresholds=cast(PromotionThresholds, SimpleNamespace()),
        current_prod_model_info=current_prod_model_info,
        previous_production_run_identity=PreviousProductionRunIdentity(
            experiment_id=None,
            train_run_id=None,
            eval_run_id=None,
            explain_run_id=None,
            promotion_id=None,
        ),
        git_commit=git_commit,
        threshold_comparison=threshold_comparison,
    )


def test_production_strategy_raises_when_runners_metadata_missing() -> None:
    """Reject production decision flow when required runners metadata is absent."""
    strategy = ProductionPromotionStrategy()
    state = cast(PromotionState, SimpleNamespace())
    context = _context(runners_metadata=None)

    with pytest.raises(RuntimeMLError, match="Runners metadata is required"):
        strategy.execute(context, state)


def test_production_strategy_promotes_and_builds_run_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Promote when thresholds pass and model beats production baseline, including run-info payload."""
    strategy = ProductionPromotionStrategy()

    threshold_comparison = ThresholdComparisonResult(
        meets_thresholds=True,
        message="all thresholds met",
        target_sets=[MetricSet.VAL],
        target_metrics=[MetricName.ROC_AUC],
        directions={MetricName.ROC_AUC: Direction.MAXIMIZE},
    )
    production_comparison = SimpleNamespace(
        beats_previous=True,
        message="beats production",
        previous_production_metrics={"val": {"auc": 0.79}},
    )

    state = _state(
        threshold_comparison=threshold_comparison,
        evaluation_metrics={"val": {"auc": 0.83}},
        current_prod_model_info={"promotion_id": "prom_123"},
        git_commit="abc123",
    )
    context = _context(runners_metadata=_runners_metadata())

    monkeypatch.setattr(
        "ml.promotion.strategies.production.compare_against_production_model",
        lambda **kwargs: production_comparison,
    )

    captured_prepare: dict[str, Any] = {}

    def _fake_prepare_run_information(**kwargs: Any) -> dict[str, Any]:
        captured_prepare.update(kwargs)
        return {"promotion_id": "new_promo_1"}

    monkeypatch.setattr(
        "ml.promotion.strategies.production.prepare_run_information",
        _fake_prepare_run_information,
    )

    result = strategy.execute(context, state)

    assert isinstance(result, PromotionResult)
    assert result.promotion_decision is True
    assert result.beats_previous is True
    assert result.run_info == {"promotion_id": "new_promo_1"}
    assert result.previous_production_metrics == {"val": {"auc": 0.79}}
    assert captured_prepare["experiment_id"] == "exp_1"
    assert captured_prepare["train_run_id"] == "train_1"
    assert captured_prepare["eval_run_id"] == "eval_1"
    assert captured_prepare["explain_run_id"] == "explain_1"
    assert captured_prepare["run_id"] == "promo_run_1"
    assert captured_prepare["timestamp"] == "20260306T210000"
    assert captured_prepare["metrics"] == {"val": {"auc": 0.83}}


def test_production_strategy_declines_without_run_info_when_criteria_not_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not create run-info payload when either threshold or production comparison fails."""
    strategy = ProductionPromotionStrategy()

    threshold_comparison = ThresholdComparisonResult(
        meets_thresholds=False,
        message="threshold miss",
        target_sets=[MetricSet.VAL],
        target_metrics=[MetricName.ROC_AUC],
        directions={MetricName.ROC_AUC: Direction.MAXIMIZE},
    )
    production_comparison = SimpleNamespace(
        beats_previous=False,
        message="does not beat production",
        previous_production_metrics={"val": {"auc": 0.85}},
    )

    state = _state(
        threshold_comparison=threshold_comparison,
        evaluation_metrics={"val": {"auc": 0.80}},
        current_prod_model_info={"promotion_id": "prom_999"},
        git_commit="def456",
    )
    context = _context(runners_metadata=_runners_metadata())

    monkeypatch.setattr(
        "ml.promotion.strategies.production.compare_against_production_model",
        lambda **kwargs: production_comparison,
    )

    called_prepare = {"called": False}

    def _unexpected_prepare(**kwargs: Any) -> dict[str, Any]:
        called_prepare["called"] = True
        return {"promotion_id": "unexpected"}

    monkeypatch.setattr(
        "ml.promotion.strategies.production.prepare_run_information",
        _unexpected_prepare,
    )

    result = strategy.execute(context, state)

    assert result.promotion_decision is False
    assert result.beats_previous is False
    assert result.run_info is None
    assert result.previous_production_metrics == {"val": {"auc": 0.85}}
    assert called_prepare["called"] is False


def test_staging_strategy_promotes_when_thresholds_met(monkeypatch: pytest.MonkeyPatch) -> None:
    """Promote to staging solely from threshold pass and build run-info payload."""
    strategy = StagingPromotionStrategy()

    threshold_comparison = ThresholdComparisonResult(
        meets_thresholds=True,
        message="threshold met",
        target_sets=[MetricSet.VAL],
        target_metrics=[MetricName.ROC_AUC],
        directions={MetricName.ROC_AUC: Direction.MAXIMIZE},
    )

    state = _state(
        threshold_comparison=threshold_comparison,
        evaluation_metrics={"val": {"auc": 0.84}},
        git_commit="git789",
    )
    context = _context(runners_metadata=_runners_metadata())

    monkeypatch.setattr(
        "ml.promotion.strategies.staging.prepare_run_information",
        lambda **kwargs: {"promotion_id": "stage_promo_1"},
    )

    result = strategy.execute(context, state)

    assert result.promotion_decision is True
    assert result.beats_previous is False
    assert result.run_info == {"promotion_id": "stage_promo_1"}
    assert result.previous_production_metrics is None
    assert result.production_comparison is None


def test_staging_strategy_raises_when_runners_metadata_missing() -> None:
    """Reject staging decision flow when required runners metadata is absent."""
    strategy = StagingPromotionStrategy()
    state = _state(
        threshold_comparison=ThresholdComparisonResult(
            meets_thresholds=True,
            message="ok",
            target_sets=[MetricSet.VAL],
            target_metrics=[MetricName.ROC_AUC],
            directions={MetricName.ROC_AUC: Direction.MAXIMIZE},
        ),
        evaluation_metrics={"val": {"auc": 0.9}},
        git_commit="zzz",
    )
    context = _context(runners_metadata=None)

    with pytest.raises(RuntimeMLError, match="Runners metadata is required"):
        strategy.execute(context, state)
