from pathlib import Path

import pytest


@pytest.fixture
def dummy_models_config(tmp_path: Path) -> dict:
    """Provide a dummy models configuration for testing."""
    return {
        "dummy_model_v1": {
            "name": "dummy_model",
            "version": "v1",
            "task": "binary_classification",
            "target": "label",
            "algorithm": "catboost",
            "features": {
                "version": "v1",
                "path": "data/features/dummy_model/v1",
                "schema": "data/features/dummy_model/v1/schema.csv",
            },
            "artifacts": {
                "model": "ml/models/trained/dummy_model_v1.joblib",
                "metadata": "ml/models/metadata/dummy_model_v1.json",
                "feature_importances": "ml/models/explainability/dummy_model_v1/feature_importances.csv",
                "shap_importances": "ml/models/explainability/dummy_model_v1/shap_importances.csv",
            },
            "explainability": {
                "feature_importance_method": "PredictionValuesChange",
                "shap_method": "tree_explainer_mean_abs",
            },
            "threshold": 0.5,
        }
    }