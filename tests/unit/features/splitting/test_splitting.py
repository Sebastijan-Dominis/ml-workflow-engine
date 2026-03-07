"""Unit tests for dataset splitting helpers."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, cast

import pandas as pd
import pytest
from ml.config.schemas.model_specs import SplitConfig, TaskConfig, TaskType

pytestmark = pytest.mark.unit

# `ml.types` imports catboost model classes at import time; provide a lightweight stub.
if "catboost" not in sys.modules:
    catboost_stub = cast(Any, types.ModuleType("catboost"))
    catboost_stub.CatBoostClassifier = type("CatBoostClassifier", (), {})
    catboost_stub.CatBoostRegressor = type("CatBoostRegressor", (), {})
    sys.modules["catboost"] = catboost_stub

splitting_module = importlib.import_module("ml.features.splitting.splitting")


def _split_cfg(*, stratify_by: str | None) -> SplitConfig:
    """Create a valid split config for tabular split tests."""
    return SplitConfig(
        strategy="random",
        stratify_by=stratify_by,
        test_size=0.2,
        val_size=0.1,
        random_state=7,
    )


def test_split_data_passes_target_series_for_stratification_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward ``y`` as stratify input when stratification is configured."""
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    cfg = _split_cfg(stratify_by="target")

    captured: dict[str, object] = {}

    def _fake_random_split(
        X_in: pd.DataFrame,
        y_in: pd.Series,
        test_size: float,
        random_state: int,
        stratify: pd.Series | None,
    ):
        captured["X"] = X_in
        captured["y"] = y_in
        captured["test_size"] = test_size
        captured["random_state"] = random_state
        captured["stratify"] = stratify
        return X_in.iloc[:2], X_in.iloc[2:], y_in.iloc[:2], y_in.iloc[2:]

    monkeypatch.setattr(splitting_module, "random_split", _fake_random_split)

    splitting_module.split_data(X, y, cfg, test_size=0.5)

    assert captured["X"] is X
    assert captured["y"] is y
    assert captured["test_size"] == 0.5
    assert captured["random_state"] == 7
    assert cast(pd.Series, captured["stratify"]).equals(y)


def test_split_data_omits_stratification_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass ``None`` for stratify when split config has no stratification target."""
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    cfg = _split_cfg(stratify_by=None)

    captured: dict[str, object] = {}

    def _fake_random_split(
        X_in: pd.DataFrame,
        y_in: pd.Series,
        test_size: float,
        random_state: int,
        stratify: pd.Series | None,
    ):
        captured["stratify"] = stratify
        return X_in.iloc[:2], X_in.iloc[2:], y_in.iloc[:2], y_in.iloc[2:]

    monkeypatch.setattr(splitting_module, "random_split", _fake_random_split)

    splitting_module.split_data(X, y, cfg, test_size=0.5)

    assert captured["stratify"] is None


def test_get_splits_tabular_returns_expected_partition_sizes_and_classification_stats() -> None:
    """Produce train/val/test splits with row counts and classification metrics populated."""
    X = pd.DataFrame({"x": list(range(100))})
    y = pd.Series(([0] * 70) + ([1] * 30))

    splits, info = splitting_module.get_splits_tabular(
        X,
        y,
        split_cfg=_split_cfg(stratify_by="target"),
        task_cfg=TaskConfig(type=TaskType.classification),
    )

    assert len(splits.X_train) == 70
    assert len(splits.X_val) == 10
    assert len(splits.X_test) == 20

    assert info.train.n_rows == 70
    assert info.val.n_rows == 10
    assert info.test.n_rows == 20

    assert info.train.class_distribution is not None
    assert info.val.class_distribution is not None
    assert info.test.class_distribution is not None
    assert info.train.positive_rate is not None
    assert info.val.positive_rate is not None
    assert info.test.positive_rate is not None


def test_get_splits_tabular_for_regression_leaves_classification_stats_empty() -> None:
    """Keep classification-only split stats unset for non-classification tasks."""
    X = pd.DataFrame({"x": list(range(20))})
    y = pd.Series([float(i) for i in range(20)])

    _, info = splitting_module.get_splits_tabular(
        X,
        y,
        split_cfg=_split_cfg(stratify_by=None),
        task_cfg=TaskConfig(type=TaskType.regression),
    )

    assert info.train.class_distribution is None
    assert info.val.class_distribution is None
    assert info.test.class_distribution is None
    assert info.train.positive_rate is None
    assert info.val.positive_rate is None
    assert info.test.positive_rate is None


def test_get_splits_raises_not_implemented_for_time_series_data_type() -> None:
    """Raise explicit not-implemented error for time-series split routing."""
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([10, 11, 12, 13])

    with pytest.raises(NotImplementedError, match="Time-series split not implemented yet"):
        splitting_module.get_splits(
            X,
            y,
            split_cfg=_split_cfg(stratify_by=None),
            data_type="time-series",
            task_cfg=TaskConfig(type=TaskType.regression),
        )


def test_get_splits_routes_tabular_to_get_splits_tabular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate tabular data type requests to tabular split implementation."""
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    cfg = _split_cfg(stratify_by=None)
    task_cfg = TaskConfig(type=TaskType.classification)

    expected_splits = object()
    expected_info = object()
    captured: dict[str, object] = {}

    def _fake_get_splits_tabular(
        X_in: pd.DataFrame,
        y_in: pd.Series,
        *,
        split_cfg: SplitConfig,
        task_cfg: TaskConfig,
    ) -> tuple[object, object]:
        captured["X"] = X_in
        captured["y"] = y_in
        captured["split_cfg"] = split_cfg
        captured["task_cfg"] = task_cfg
        return expected_splits, expected_info

    monkeypatch.setattr(splitting_module, "get_splits_tabular", _fake_get_splits_tabular)

    splits, info = splitting_module.get_splits(
        X,
        y,
        split_cfg=cfg,
        data_type="tabular",
        task_cfg=task_cfg,
    )

    assert splits is expected_splits
    assert info is expected_info
    assert captured["X"] is X
    assert captured["y"] is y
    assert captured["split_cfg"] is cfg
    assert captured["task_cfg"] is task_cfg


def test_get_splits_raises_for_unsupported_data_type_value() -> None:
    """Raise explicit error when data type falls outside supported routing values."""
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([10, 11, 12, 13])

    with pytest.raises(NotImplementedError, match="Unsupported data_type: image"):
        splitting_module.get_splits(
            X,
            y,
            split_cfg=_split_cfg(stratify_by=None),
            data_type=cast(Any, "image"),
            task_cfg=TaskConfig(type=TaskType.regression),
        )
