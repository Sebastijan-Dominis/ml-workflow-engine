"""Unit tests for best-F1 threshold selection helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ml.runners.training.utils.metrics.best_f1 import get_best_f1_threshold

pytestmark = pytest.mark.unit


class _PipelineWithTrackedPredictProba:
    """Pipeline stub exposing deterministic class-1 probabilities."""

    def __init__(self, probs_class_1: list[float]) -> None:
        self._probs_class_1 = probs_class_1
        self.calls = 0
        self.received_x: pd.DataFrame | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return two-column probabilities while tracking invocation details."""
        self.calls += 1
        self.received_x = X
        p1 = np.asarray(self._probs_class_1, dtype=float)
        return np.column_stack((1.0 - p1, p1))


def test_get_best_f1_threshold_returns_expected_optimal_threshold_and_score() -> None:
    """Find exact optimum from the fixed 0.01 threshold grid used by helper."""
    pipeline = _PipelineWithTrackedPredictProba([0.10, 0.40, 0.60, 0.90])
    X = pd.DataFrame({"f": [1, 2, 3, 4]})
    y_true = pd.Series([0, 0, 1, 1])

    threshold, best_f1 = get_best_f1_threshold(pipeline, X, y_true)

    assert threshold == pytest.approx(0.41)
    assert best_f1 == pytest.approx(1.0)
    assert pipeline.calls == 1
    assert pipeline.received_x is X


def test_get_best_f1_threshold_uses_first_threshold_when_scores_tie() -> None:
    """Prefer the earliest threshold when multiple thresholds share max F1."""
    pipeline = _PipelineWithTrackedPredictProba([0.90, 0.90, 0.90, 0.90])
    X = pd.DataFrame({"f": [5, 6, 7, 8]})
    y_true = pd.Series([1, 1, 1, 1])

    threshold, best_f1 = get_best_f1_threshold(pipeline, X, y_true)

    # Every threshold <= 0.90 yields perfect F1, so argmax should pick 0.00.
    assert threshold == pytest.approx(0.0)
    assert best_f1 == pytest.approx(1.0)


def test_get_best_f1_threshold_returns_builtin_float_types() -> None:
    """Return plain Python floats for downstream JSON-serialization compatibility."""
    pipeline = _PipelineWithTrackedPredictProba([0.20, 0.80])
    X = pd.DataFrame({"f": [1, 2]})
    y_true = pd.Series([0, 1])

    threshold, best_f1 = get_best_f1_threshold(pipeline, X, y_true)

    assert isinstance(threshold, float)
    assert isinstance(best_f1, float)
