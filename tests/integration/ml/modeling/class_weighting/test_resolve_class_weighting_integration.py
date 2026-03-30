from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.modeling.class_weighting.models import DataStats
from ml.modeling.class_weighting.resolve_class_weighting import resolve_class_weighting

pytestmark = pytest.mark.integration


def _make_config(policy: str, strategy: str | None = None, imbalance_threshold: float | None = None):
    return SimpleNamespace(class_weighting=SimpleNamespace(policy=policy, strategy=strategy, imbalance_threshold=imbalance_threshold))


def test_resolve_class_weighting_ratio_for_catboost_and_xgboost() -> None:
    config = _make_config(policy="ratio", strategy="ratio")
    stats = DataStats(n_samples=12, class_counts={0: 10, 1: 2}, minority_ratio=2 / 12)

    res_cb = resolve_class_weighting(cast(Any, config), stats, library="catboost")
    assert "class_weights" in res_cb and isinstance(res_cb["class_weights"], list)
    assert pytest.approx(res_cb["class_weights"][1]) == 10 / 2

    res_xgb = resolve_class_weighting(cast(Any, config), stats, library="xgboost")
    assert "scale_pos_weight" in res_xgb
    assert pytest.approx(res_xgb["scale_pos_weight"]) == 10 / 2
