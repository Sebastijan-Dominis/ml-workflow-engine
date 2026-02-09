"""Integration and unit tests for the evaluation CLI and helpers.

Exercises argument parsing, config loading, threshold validation and
the top-level `main` evaluation flow for classification models.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ml.runners.evaluation import evaluate
from ml.runners.evaluation.evaluate import get_model_configs, main, parse_args, validate_threshold


def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI argument parsing should populate args correctly."""

    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--problem", "cancellation", "--segment", "global",
         "--version", "v1", "--experiment-id", "20260101_000000_abc12345"],
    )
    args = parse_args()
    assert args.problem == "cancellation"
    assert args.segment == "global"
    assert args.version == "v1"
    assert args.experiment_id == "20260101_000000_abc12345"

def test_get_model_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dummy_models_config) -> None:
    """Model config loader reads `configs/models.yaml` and returns mapping."""

    # Create expected path structure under tmp
    base = tmp_path / "configs"
    base.mkdir(parents=True)
    cfg_file = base / "models.yaml"
    cfg_file.write_text(yaml.safe_dump(dummy_models_config))

    # Run with cwd at tmp_path so relative path resolves
    monkeypatch.chdir(tmp_path)
    loaded = get_model_configs("dummy_model_v1")
    assert loaded["name"] == "dummy_model"

def test_validate_threshold() -> None:
    """Acceptable threshold values [0.0, 1.0] should pass, others raise."""

    # Valid thresholds should not raise
    validate_threshold(0.0)
    validate_threshold(0.5)
    validate_threshold(1.0)

    # Invalid thresholds should raise
    with pytest.raises(ValueError):
        validate_threshold(-0.1)
    with pytest.raises(ValueError):
        validate_threshold(1.1)

def test_unsupported_task(monkeypatch: pytest.MonkeyPatch, dummy_models_config) -> None:
    """Main should raise ValueError when a model config lists an unsupported task."""

    # Setup monkeypatches
    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: type("A", (), {
            "problem": "dummy",
            "segment": "model",
            "version": "v1",
            "experiment_id": "20260101_000000_abc12345",
            "logging_level": "INFO",
        })()
    )
    dummy_models_config["dummy_model_v1"]["task"] = "unsupported_task"
    monkeypatch.setattr(
        evaluate,
        "get_model_configs",
        lambda _: dummy_models_config["dummy_model_v1"]
    )

    # Mock setup_logging to do nothing
    monkeypatch.setattr(evaluate, "setup_logging", lambda *a, **kw: None)

    result = main()
    assert result != 0

@pytest.mark.integration
def test_evaluate_main_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dummy_models_config: dict) -> None:
    """Smoke: run `main()` end-to-end with dummy model and feature data.

    This test writes minimal feature files and a serialised dummy
    pipeline and asserts the `main()` call completes without error.
    """

    # Create expected path structure under tmp
    base = tmp_path / "configs"
    base.mkdir(parents=True)
    cfg_file = base / "models.yaml"
    cfg_file.write_text(yaml.safe_dump(dummy_models_config))


    # Create dummy feature and label data
    X_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [3, 4, 5, 6, 7]})
    y_train = pd.DataFrame([0, 1, 0, 1, 0])
    X_val = pd.DataFrame({"feature1": [5, 6, 7], "feature2": [6, 7, 8]})
    y_val = pd.DataFrame([1, 0, 1])
    X_test = pd.DataFrame({"feature1": [7, 8, 9], "feature2": [8, 9, 10]})
    y_test = pd.DataFrame([0, 1, 0])

    # Create and fit dummy model
    pipeline = Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    pipeline.fit(X_train, y_train.values.ravel())
    model_file = tmp_path / "ml/models/trained/dummy_model_v1.joblib"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_file)

    # Save feature data to expected location
    features_path = tmp_path / "data/features/dummy_model/v1"
    features_path.mkdir(parents=True, exist_ok=True)

    # Save feature and label data
    X_train.to_parquet(features_path / "X_train.parquet")
    y_train.to_parquet(features_path / "y_train.parquet")
    X_val.to_parquet(features_path / "X_val.parquet")
    y_val.to_parquet(features_path / "y_val.parquet")
    X_test.to_parquet(features_path / "X_test.parquet")
    y_test.to_parquet(features_path / "y_test.parquet")

    # Create dummy metadata file
    metadata = {
        "name": "dummy_model",
        "version": "v1",
        "task": "binary_classification",
        "target": "label",
        "algorithm": "dummy",
    }
    metadata_file = tmp_path / "ml/models/metadata/dummy_model_v1.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    # Run main with monkeypatched argv
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--problem", "dummy", "--segment", "model",
         "--version", "v1", "--experiment-id", "20260101_000000_abc12345"],
    )
    monkeypatch.chdir(tmp_path)
    # create configs/models.yaml, model file, metadata file
    main()
    # If no exceptions are raised, the test passes
    assert True
