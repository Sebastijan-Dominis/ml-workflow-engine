"""Unit tests for randomized search orchestration utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from ml.config.schemas.model_cfg import SearchModelConfig
from ml.search.utils import randomized_search as randomized_search_module
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

pytestmark = pytest.mark.unit


class _ResolvedCV:
    """Minimal CV object exposing `get_n_splits` for logging-path coverage."""

    def __init__(self, n_splits: int) -> None:
        self._n_splits = n_splits

    def get_n_splits(self, x: pd.DataFrame, y: pd.Series) -> int:
        """Return a fixed split count expected by the caller."""
        _ = (x, y)
        return self._n_splits


class _FakeRandomizedSearchCV:
    """Test double for capturing init kwargs and returning stable best-result fields."""

    last_init_kwargs: dict[str, Any] | None = None
    fit_x: pd.DataFrame | None = None
    fit_y: pd.Series | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.__class__.last_init_kwargs = kwargs
        self.best_params_ = {"Model__depth": 6}
        self.best_score_ = 0.8125
        self.best_index_ = np.int64(2)
        self.cv_results_ = {
            "mean_test_score": np.array([0.7, 0.8, 0.8125]),
            "std_test_score": np.array([0.02, 0.01, 0.03]),
            "rank_test_score": np.array([3, 2, 1]),
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Capture fit inputs to assert that caller forwards training data unchanged."""
        self.__class__.fit_x = x
        self.__class__.fit_y = y


def test_perform_randomized_search_uses_gpu_safe_defaults_and_serializes_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force `n_jobs=1` on GPU and return list-serialized CV fields with default knobs."""
    model_cfg = cast(
        SearchModelConfig,
        SimpleNamespace(
            cv=3,
            verbose=None,
            search=SimpleNamespace(
                broad=SimpleNamespace(n_iter=4),
                random_state=7,
                error_score=None,
                hardware=SimpleNamespace(task_type=SimpleNamespace(value="GPU")),
            ),
        ),
    )
    captured_classifier_flags: list[bool] = []

    monkeypatch.setattr(randomized_search_module, "RandomizedSearchCV", _FakeRandomizedSearchCV)
    monkeypatch.setattr(
        randomized_search_module,
        "check_cv",
        lambda cv, y, classifier: (
            captured_classifier_flags.append(classifier),
            _ResolvedCV(3),
        )[1],
    )
    monkeypatch.setattr(randomized_search_module, "is_classifier", lambda _: True)

    pipeline = Pipeline(steps=[("identity", FunctionTransformer(validate=False))])
    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 1, 0])
    param_distributions = {"Model__depth": [4, 6, 8]}

    result = randomized_search_module.perform_randomized_search(
        pipeline,
        X_train=X_train,
        y_train=y_train,
        param_distributions=param_distributions,
        model_cfg=model_cfg,
        search_phase="broad",
        scoring="roc_auc",
    )

    assert _FakeRandomizedSearchCV.last_init_kwargs is not None
    assert _FakeRandomizedSearchCV.last_init_kwargs["n_jobs"] == 1
    assert _FakeRandomizedSearchCV.last_init_kwargs["verbose"] == 100
    assert np.isnan(_FakeRandomizedSearchCV.last_init_kwargs["error_score"])
    assert _FakeRandomizedSearchCV.fit_x is X_train
    assert _FakeRandomizedSearchCV.fit_y is y_train
    assert captured_classifier_flags == [True]

    assert result["best_params"] == {"Model__depth": 6}
    assert result["best_index"] == 2
    assert isinstance(result["best_index"], int)
    assert result["cv_results"]["mean_test_score"] == [0.7, 0.8, 0.8125]
    assert result["cv_results"]["rank_test_score"] == [3, 2, 1]
    assert result["cv"] == 3
    assert result["error_score"] == "nan"
    assert result["search_phase"] == "broad"


def test_perform_randomized_search_respects_cpu_jobs_and_non_int_cv_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use `n_jobs=-1` on CPU and report non-integer CV type by class name."""

    class _CustomCV:
        pass

    cv_obj = _CustomCV()
    model_cfg = cast(
        SearchModelConfig,
        SimpleNamespace(
            cv=cv_obj,
            verbose=0,
            search=SimpleNamespace(
                narrow=SimpleNamespace(n_iter=2),
                random_state=11,
                error_score="raise",
                hardware=SimpleNamespace(task_type=SimpleNamespace(value="CPU")),
            ),
        ),
    )

    monkeypatch.setattr(randomized_search_module, "RandomizedSearchCV", _FakeRandomizedSearchCV)
    captured_classifier_flags: list[bool] = []
    monkeypatch.setattr(
        randomized_search_module,
        "check_cv",
        lambda cv, y, classifier: (
            captured_classifier_flags.append(classifier),
            _ResolvedCV(4),
        )[1],
    )
    monkeypatch.setattr(randomized_search_module, "is_classifier", lambda _: False)

    pipeline = Pipeline(steps=[("identity", FunctionTransformer(validate=False))])
    X_train = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0]})
    y_train = pd.Series([1, 0, 1, 0])

    result = randomized_search_module.perform_randomized_search(
        pipeline,
        X_train=X_train,
        y_train=y_train,
        param_distributions={"Model__l2_leaf_reg": [1.0, 3.0]},
        model_cfg=model_cfg,
        search_phase="narrow",
        scoring="roc_auc",
    )

    assert _FakeRandomizedSearchCV.last_init_kwargs is not None
    assert _FakeRandomizedSearchCV.last_init_kwargs["n_jobs"] == -1
    assert _FakeRandomizedSearchCV.last_init_kwargs["cv"] is cv_obj
    assert _FakeRandomizedSearchCV.last_init_kwargs["verbose"] == 0
    assert _FakeRandomizedSearchCV.last_init_kwargs["error_score"] == "raise"
    assert captured_classifier_flags == [False]

    assert result["n_iter"] == 2
    assert result["cv"] == "_CustomCV"
    assert result["random_state"] == 11
    assert result["search_phase"] == "narrow"


def test_perform_randomized_search_serializes_none_cv_as_non_int_class_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialize `cv=None` using class-name path while still using resolved CV for splits."""
    model_cfg = cast(
        SearchModelConfig,
        SimpleNamespace(
            cv=None,
            verbose=1,
            search=SimpleNamespace(
                broad=SimpleNamespace(n_iter=1),
                random_state=123,
                error_score=0.0,
                hardware=SimpleNamespace(task_type=SimpleNamespace(value="CPU")),
            ),
        ),
    )

    monkeypatch.setattr(randomized_search_module, "RandomizedSearchCV", _FakeRandomizedSearchCV)
    monkeypatch.setattr(randomized_search_module, "check_cv", lambda cv, y, classifier: _ResolvedCV(5))
    monkeypatch.setattr(randomized_search_module, "is_classifier", lambda _: True)

    pipeline = Pipeline(steps=[("identity", FunctionTransformer(validate=False))])
    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_train = pd.Series([0, 1, 0, 1, 0])

    result = randomized_search_module.perform_randomized_search(
        pipeline,
        X_train=X_train,
        y_train=y_train,
        param_distributions={"Model__depth": [6]},
        model_cfg=model_cfg,
        search_phase="broad",
        scoring="roc_auc",
    )

    assert _FakeRandomizedSearchCV.last_init_kwargs is not None
    assert _FakeRandomizedSearchCV.last_init_kwargs["cv"] is None
    assert _FakeRandomizedSearchCV.last_init_kwargs["error_score"] == 0.0
    assert result["cv"] == "NoneType"
    assert result["error_score"] == "0.0"
