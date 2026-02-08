"""Unit tests for CatBoost explainability helper functions.

These tests exercise helpers that inspect pipelines, compute feature
names and SHAP importances, and orchestrate the explainability flow.
"""

import pytest
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import FunctionTransformer
from catboost import CatBoost

# Commented out imports are for potential future tests
from ml.runners.explainability.custom_explainability_scripts.explain_catboost import (
    # import_components,
    check_key_presence,
    # get_pipeline_and_model,
    inspect_pipeline,
    get_feature_names,
    validate_lengths,
    # get_feature_importances,
    # get_test_data,
    get_shap_importances,
    save_importances,
    explain_catboost
)

def test_check_key_presence_missing_key() -> None:
    """Expect KeyError when required explainability keys are absent."""

    cfg = {"name": "dummy_model", "version": "v1"}
    with pytest.raises(KeyError):
        check_key_presence(cfg)

def test_inspect_pipeline_valid() -> None:
    """Passes when pipeline contains a final model estimator."""

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DummyClassifier())
    ])
    inspect_pipeline(pipeline)
    # If no exception is raised, the test passes

def test_inspect_pipeline_invalid_last_step() -> None:
    """Raises ValueError when the pipeline has no final model step."""

    pipe = Pipeline([("scaler", StandardScaler())])

    with pytest.raises(ValueError):
        inspect_pipeline(pipe)

def test_get_feature_names_returns_columns() -> None:
    """Return original DataFrame columns when transformer preserves columns."""

    X = pd.DataFrame({"a": [1], "b": [2]})

    pipe = Pipeline([
        ("identity", FunctionTransformer(lambda x: x)),
        ("model", object())
    ])

    names = get_feature_names(pipe, X)
    assert set(names) == {"a", "b"}

def test_validate_lengths_mismatch() -> None:
    """Ensure length mismatch between features and importances raises."""

    with pytest.raises(ValueError):
        validate_lengths(np.array(["a", "b"]), np.array([1.0]))

def test_get_shap_importances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compute SHAP importances using a mocked TreeExplainer implementation."""

    class DummyExplainer:
        def shap_values(self, X):
            return np.ones((X.shape[0], X.shape[1]))

    monkeypatch.setattr(
        "ml.training.explain_scripts.custom_explainability_scripts.explain_catboost.shap.TreeExplainer",
        lambda *a, **k: DummyExplainer()
    )

    X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    feature_names = np.array(["f1", "f2"])

    pipeline = Pipeline([
        ('identity', FunctionTransformer(lambda X: pd.DataFrame(X, columns=X.columns))),
        ('model', DummyClassifier())
    ])

    X_train = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    y_train = pd.Series([0, 1])

    pipeline.fit(X_train, y_train)

    cfg = {"explainability": {"shap_method": "tree_explainer_mean_abs"}}

    df = get_shap_importances(feature_names, CatBoost(), pipeline, X, cfg)
    assert "mean_abs_shap" in df.columns


def test_get_shap_importances_invalid_method() -> None:
    """Invalid `shap_method` value should raise a ValueError."""

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DummyClassifier())
    ])
    with pytest.raises(ValueError):
        get_shap_importances(
            np.array(["a"]),
            CatBoost(),
            pipeline,
            pd.DataFrame({"a": [1]}),
            {"explainability": {"shap_method": "invalid"}}
        )


def test_save_importances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persist computed importances CSV under the model explainability folder."""

    monkeypatch.chdir(tmp_path)

    df = pd.DataFrame({"feature": ["a"], "importance": [1.0]})
    save_importances(df, "dummy_model_v1", "imp.csv", "Feature")

    out = tmp_path / "ml/models/explainability/dummy_model_v1/imp.csv"
    assert out.exists()

@pytest.mark.integration
def test_explain_catboost_orchestration(monkeypatch: pytest.MonkeyPatch) -> None:
    """High-level orchestration: exercise `explain_catboost` with helpers mocked.

    All I/O and heavy compute functions are replaced with shallow
    stand-ins to validate the control flow.
    """

    monkeypatch.setattr("ml.training.explain_scripts.custom_explainability_scripts.explain_catboost.get_pipeline_and_model", lambda _: ("model", "pipeline"))
    monkeypatch.setattr("ml.training.explain_scripts.custom_explainability_scripts.explain_catboost.inspect_pipeline", lambda _: None)
    monkeypatch.setattr("ml.training.explain_scripts.custom_explainability_scripts.explain_catboost.get_test_data", lambda _: None)
    monkeypatch.setattr("ml.training.explain_scripts.custom_explainability_scripts.explain_catboost.get_feature_names", lambda *_: ["a"])
    monkeypatch.setattr("ml.training.explain_scripts.custom_explainability_scripts.explain_catboost.get_feature_importances", lambda *_: None)
    monkeypatch.setattr("ml.training.explain_scripts.custom_explainability_scripts.explain_catboost.get_shap_importances", lambda *_: None)
    monkeypatch.setattr("ml.training.explain_scripts.custom_explainability_scripts.explain_catboost.save_importances", lambda *_: None)

    cfg = {
        "name": "dummy_model",
        "version": "v1",
        "features": {"path": "."},
        "explainability": {}
    }

    explain_catboost(cfg)
    # If no exception is raised, the test passes
