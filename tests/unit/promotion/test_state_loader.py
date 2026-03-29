"""Unit tests for promotion state loader orchestration logic."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.promotion.constants.constants import PreviousProductionRunIdentity
from ml.promotion.context import PromotionContext
from ml.promotion.state_loader import PromotionStateLoader

pytestmark = pytest.mark.unit


def _context(*, problem: str = "cancellation", segment: str = "city_hotel") -> PromotionContext:
    """Build a minimal promotion context object required by the state loader."""
    paths = SimpleNamespace(
        registry_path=Path("model_registry") / "models.yaml",
        archive_path=Path("model_registry") / "archive.yaml",
        promotion_configs_dir=Path("configs") / "promotion",
        eval_run_dir=Path("experiments") / "eval" / "run-1",
    )
    args = SimpleNamespace(problem=problem, segment=segment)
    return cast(PromotionContext, SimpleNamespace(paths=paths, args=args))


def test_state_loader_load_builds_full_state_with_production_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate state fields from loader dependencies and derive previous production run identity."""
    model_registry = {
        "cancellation": {
            "city_hotel": {
                "production": {
                    "experiment_id": "exp-10",
                    "train_run_id": "train-10",
                    "eval_run_id": "eval-10",
                    "explain_run_id": "explain-10",
                    "promotion_id": "prom-10",
                    "metrics": {"val": {"f1": 0.79}},
                }
            }
        }
    }
    archive_registry: dict[str, Any] = {"archive": []}
    global_thresholds = {"raw": "thresholds"}
    metrics_file = {"metrics": {"val": {"f1": 0.82}}}

    thresholds_raw = {"threshold": "raw"}
    validated_thresholds = cast(Any, SimpleNamespace(name="validated-thresholds"))
    threshold_comparison = cast(Any, SimpleNamespace(meets_thresholds=True, message="ok"))

    def _load_yaml(path: Path) -> dict:
        if str(path).endswith("models.yaml"):
            return model_registry
        if str(path).endswith("archive.yaml"):
            return archive_registry
        if str(path).endswith("thresholds.yaml"):
            return global_thresholds
        raise AssertionError(f"Unexpected YAML path: {path}")

    compare_calls: list[tuple[dict[str, dict[str, float]], Any]] = []

    def _compare_against_thresholds(*, evaluation_metrics: dict[str, dict[str, float]], promotion_thresholds: Any) -> Any:
        compare_calls.append((evaluation_metrics, promotion_thresholds))
        return threshold_comparison

    monkeypatch.setattr("ml.promotion.state_loader.load_yaml", _load_yaml)
    monkeypatch.setattr("ml.promotion.state_loader.load_json", lambda _: metrics_file)
    monkeypatch.setattr("ml.promotion.state_loader.extract_thresholds", lambda **kwargs: thresholds_raw)
    monkeypatch.setattr("ml.promotion.state_loader.validate_promotion_thresholds", lambda _: validated_thresholds)
    monkeypatch.setattr("ml.promotion.state_loader.get_git_commit", lambda: "commit-abc123")
    monkeypatch.setattr("ml.promotion.state_loader.compare_against_thresholds", _compare_against_thresholds)

    state = PromotionStateLoader().load(_context())

    assert state.model_registry == model_registry
    assert state.archive_registry == archive_registry
    assert state.evaluation_metrics == {"val": {"f1": 0.82}}
    assert state.promotion_thresholds is validated_thresholds
    assert state.current_prod_model_info == model_registry["cancellation"]["city_hotel"]["production"]
    assert state.git_commit == "commit-abc123"
    assert state.threshold_comparison is threshold_comparison

    assert state.previous_production_run_identity == PreviousProductionRunIdentity(
        experiment_id="exp-10",
        train_run_id="train-10",
        eval_run_id="eval-10",
        explain_run_id="explain-10",
        promotion_id="prom-10",
    )
    assert compare_calls == [({"val": {"f1": 0.82}}, validated_thresholds)]


def test_state_loader_load_defaults_previous_identity_when_no_production_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set current production info and previous identity fields to None when registry has no production model."""
    monkeypatch.setattr("ml.promotion.state_loader.load_yaml", lambda _: {})
    monkeypatch.setattr("ml.promotion.state_loader.load_json", lambda _: {"metrics": {"val": {"f1": 0.82}}})
    monkeypatch.setattr("ml.promotion.state_loader.extract_thresholds", lambda **kwargs: {"raw": 1})
    monkeypatch.setattr("ml.promotion.state_loader.validate_promotion_thresholds", lambda _: cast(Any, SimpleNamespace()))
    monkeypatch.setattr("ml.promotion.state_loader.get_git_commit", lambda: "commit-none")
    monkeypatch.setattr(
        "ml.promotion.state_loader.compare_against_thresholds",
        lambda **kwargs: cast(Any, SimpleNamespace(meets_thresholds=True, message="ok")),
    )

    state = PromotionStateLoader().load(_context())

    assert state.current_prod_model_info is None
    assert state.previous_production_run_identity == PreviousProductionRunIdentity(
        experiment_id=None,
        train_run_id=None,
        eval_run_id=None,
        explain_run_id=None,
        promotion_id=None,
    )


def test_state_loader_load_uses_empty_metrics_when_metrics_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback to empty evaluation metrics when metrics.json lacks top-level metrics key."""
    monkeypatch.setattr("ml.promotion.state_loader.load_yaml", lambda _: {})
    monkeypatch.setattr("ml.promotion.state_loader.load_json", lambda _: {"unexpected": "payload"})
    monkeypatch.setattr("ml.promotion.state_loader.extract_thresholds", lambda **kwargs: {"raw": 1})
    monkeypatch.setattr("ml.promotion.state_loader.validate_promotion_thresholds", lambda _: cast(Any, SimpleNamespace()))
    monkeypatch.setattr("ml.promotion.state_loader.get_git_commit", lambda: "commit-empty")

    seen: dict[str, dict[str, float]] = {"not_set": {"x": 1.0}}

    def _compare(*, evaluation_metrics: dict[str, dict[str, float]], promotion_thresholds: Any) -> Any:
        seen.clear()
        seen.update(evaluation_metrics)
        return cast(Any, SimpleNamespace(meets_thresholds=True, message="ok"))

    monkeypatch.setattr("ml.promotion.state_loader.compare_against_thresholds", _compare)

    state = PromotionStateLoader().load(_context())

    assert state.evaluation_metrics == {}
    assert seen == {}
