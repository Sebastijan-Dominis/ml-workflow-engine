"""Unit tests for class-weighting and metric policy resolution logic."""

from dataclasses import dataclass
from typing import cast

import pytest
from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import ConfigError
from ml.modeling.class_weighting.constants import SUPPORTED_LIBRARIES
from ml.modeling.class_weighting.models import DataStats
from ml.modeling.class_weighting.resolve_class_weighting import resolve_class_weighting
from ml.modeling.class_weighting.resolve_metric import resolve_metric

pytestmark = pytest.mark.unit

ModelConfig = SearchModelConfig | TrainModelConfig


@dataclass
class _ScoringCfg:
    """Minimal scoring configuration stub used by policy-resolution tests."""

    policy: str
    fixed_metric: str | None = None
    pr_auc_threshold: float | None = None


@dataclass
class _ClassWeightingCfg:
    """Minimal class-weighting configuration stub used by resolver tests."""

    policy: str
    strategy: str | None = None
    imbalance_threshold: float | None = None


@dataclass
class _ConfigStub:
    """Minimal config object that mirrors resolver attribute access patterns."""

    scoring: _ScoringCfg
    class_weighting: _ClassWeightingCfg


def _as_model_config(config: _ConfigStub) -> ModelConfig:
    """Cast the lightweight test stub to the resolver's model-union contract."""
    return cast(ModelConfig, config)


def test_resolve_metric_returns_fixed_metric_for_fixed_policy() -> None:
    """Return the configured metric directly when scoring policy is fixed."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="fixed", fixed_metric="roc_auc"),
        class_weighting=_ClassWeightingCfg(policy="off"),
    )

    assert resolve_metric(_as_model_config(config), stats=None) == "roc_auc"


def test_resolve_metric_returns_default_rmse_for_regression_default_policy() -> None:
    """Use sklearn-compatible negative RMSE string for regression-default policy."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="regression_default"),
        class_weighting=_ClassWeightingCfg(policy="off"),
    )

    assert resolve_metric(_as_model_config(config), stats=None) == "neg_root_mean_squared_error"


@pytest.mark.parametrize(
    ("minority_ratio", "expected_metric"),
    [
        (0.05, "average_precision"),
        (0.20, "roc_auc"),
    ],
)
def test_resolve_metric_switches_adaptive_binary_metric_by_threshold(
    minority_ratio: float,
    expected_metric: str,
) -> None:
    """Select PR-AUC below threshold and ROC-AUC at-or-above threshold."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="adaptive_binary", pr_auc_threshold=0.20),
        class_weighting=_ClassWeightingCfg(policy="off"),
    )
    stats = DataStats(n_samples=100, class_counts={0: 80, 1: 20}, minority_ratio=minority_ratio)

    assert resolve_metric(_as_model_config(config), stats=stats) == expected_metric


def test_resolve_metric_raises_when_non_fixed_policy_has_no_stats() -> None:
    """Require dataset stats for policies that adapt to class-distribution state."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="adaptive_binary", pr_auc_threshold=0.20),
        class_weighting=_ClassWeightingCfg(policy="off"),
    )

    with pytest.raises(ConfigError, match="Stats must be provided"):
        resolve_metric(_as_model_config(config), stats=None)


def test_resolve_metric_raises_when_fixed_policy_has_no_metric() -> None:
    """Fail fast when fixed scoring policy omits its required fixed metric."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="fixed", fixed_metric=None),
        class_weighting=_ClassWeightingCfg(policy="off"),
    )

    with pytest.raises(ConfigError, match="fixed_metric must be set"):
        resolve_metric(_as_model_config(config), stats=None)


def test_resolve_class_weighting_returns_empty_when_policy_off() -> None:
    """Emit no weighting parameters when class-weighting policy is disabled."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="fixed", fixed_metric="roc_auc"),
        class_weighting=_ClassWeightingCfg(policy="off", strategy="ratio"),
    )
    stats = DataStats(n_samples=100, class_counts={0: 80, 1: 20}, minority_ratio=0.20)

    assert (
        resolve_class_weighting(_as_model_config(config), stats, library="xgboost")
        == {}
    )


def test_resolve_class_weighting_returns_empty_if_not_imbalanced_enough() -> None:
    """Skip weighting when imbalance policy threshold is not exceeded."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="fixed", fixed_metric="roc_auc"),
        class_weighting=_ClassWeightingCfg(
            policy="if_imbalanced",
            strategy="ratio",
            imbalance_threshold=0.15,
        ),
    )
    stats = DataStats(n_samples=100, class_counts={0: 80, 1: 20}, minority_ratio=0.20)

    assert (
        resolve_class_weighting(_as_model_config(config), stats, library="xgboost")
        == {}
    )


def test_resolve_class_weighting_raises_if_imbalanced_policy_missing_threshold() -> None:
    """Require imbalance threshold when using the if-imbalanced policy mode."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="fixed", fixed_metric="roc_auc"),
        class_weighting=_ClassWeightingCfg(
            policy="if_imbalanced",
            strategy="ratio",
            imbalance_threshold=None,
        ),
    )
    stats = DataStats(n_samples=100, class_counts={0: 80, 1: 20}, minority_ratio=0.20)

    with pytest.raises(ConfigError, match="imbalance_threshold must be set"):
        resolve_class_weighting(_as_model_config(config), stats, library="xgboost")


@pytest.mark.parametrize(
    ("library", "expected"),
    [
        ("xgboost", {"scale_pos_weight": 4.0}),
        ("lightgbm", {"scale_pos_weight": 4.0}),
        ("catboost", {"class_weights": [1.0, 4.0]}),
    ],
)
def test_resolve_class_weighting_ratio_strategy_returns_library_specific_params(
    library: SUPPORTED_LIBRARIES,
    expected: dict[str, list[float] | float],
) -> None:
    """Return ratio-based params mapped to each supported boosting library."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="fixed", fixed_metric="roc_auc"),
        class_weighting=_ClassWeightingCfg(policy="always", strategy="ratio"),
    )
    stats = DataStats(n_samples=100, class_counts={0: 80, 1: 20}, minority_ratio=0.20)

    assert (
        resolve_class_weighting(_as_model_config(config), stats, library=library)
        == expected
    )


def test_resolve_class_weighting_balanced_strategy_matches_sklearn_weights() -> None:
    """Generate balanced class weights equivalent to sklearn's formula."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="fixed", fixed_metric="roc_auc"),
        class_weighting=_ClassWeightingCfg(policy="always", strategy="balanced"),
    )
    stats = DataStats(n_samples=100, class_counts={0: 90, 1: 10}, minority_ratio=0.10)

    resolved = resolve_class_weighting(_as_model_config(config), stats, library="sklearn")

    assert "class_weights" in resolved
    assert resolved["class_weights"] == pytest.approx([100 / (2 * 90), 100 / (2 * 10)])


def test_resolve_class_weighting_raises_for_ratio_strategy_on_unsupported_library() -> None:
    """Raise a config error when ratio strategy is requested for unsupported backend."""
    config = _ConfigStub(
        scoring=_ScoringCfg(policy="fixed", fixed_metric="roc_auc"),
        class_weighting=_ClassWeightingCfg(policy="always", strategy="ratio"),
    )
    stats = DataStats(n_samples=100, class_counts={0: 80, 1: 20}, minority_ratio=0.20)

    with pytest.raises(ConfigError, match="Unsupported class weighting strategy"):
        resolve_class_weighting(_as_model_config(config), stats, library="sklearn")
