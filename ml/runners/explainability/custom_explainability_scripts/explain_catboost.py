"""Explainability utilities for CatBoost models.

This module contains helper functions used to produce global explainability
artifacts for trained CatBoost models: feature importances and SHAP-based
global explanations. The functions encapsulate loading the trained pipeline,
validating pipeline structure, extracting feature names, computing importances
and persisting results to disk.

The main entrypoint is `explain_catboost(model_configs)`, which orchestrates the
workflow and writes CSVs to `ml/models/explainability/<name>_<version>/`.
"""

# General imports
import importlib
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from catboost import CatBoost
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def import_components(name_and_version: str) -> None:
    """Dynamically import model-specific component modules.

    Some custom transformer/estimator objects are defined inside
    `ml.components.<name_and_version>` and must be imported prior to
    deserializing a joblib pipeline so that class definitions are available.

    Args:
        name_and_version (str): Module suffix matching the trained model,
            e.g. "cancellation_v1".
    """

    importlib.import_module(f"ml.components.{name_and_version}")

def check_key_presence(model_configs: dict) -> None:
    """Validate that required configuration keys are present.

    Args:
        model_configs (dict): Model configuration as loaded from
            `configs/models.yaml`.

    Raises:
        KeyError: If any of the required keys are missing.
    """

    required_keys = ["explainability", "features", "name", "version"]
    for k in required_keys:
        if k not in model_configs:
            msg = f"Missing required config key: {k}"
            logger.error(msg)
            raise KeyError(msg)

def get_pipeline_and_model(name_and_version: str) -> tuple:
    """Load a joblib-serialized pipeline and extract the final model step.

    This function imports any model-specific components before attempting
    deserialization so that custom classes are resolvable.

    Args:
        name_and_version (str): Base name of the joblib file (without
            extension), e.g. "cancellation_v1".

    Returns:
        tuple: `(model, pipeline)` where `pipeline` is the deserialized
            sklearn-style pipeline and `model` is the final estimator stored
            under `pipeline.named_steps['model']`.

    Raises:
        Exception: Propagates errors raised by joblib.load or if the expected
            pipeline structure is not present.
    """

    import_components(name_and_version)

    try:
        pipeline = joblib.load(f"ml/models/trained/{name_and_version}.joblib")
    except Exception:
        logger.exception(f"Error loading the trained model pipeline for {name_and_version}")
        raise

    model = pipeline.named_steps["model"]

    return model, pipeline

def inspect_pipeline(pipeline: Pipeline) -> None:
    """Validate pipeline shape and interface expectations.

    The function checks that `pipeline` exposes `named_steps`, that the
    final step is called `'model'`, and that the preprocessing portion of the
    pipeline (all steps except the final model) implements a `transform`
    method so it can be used to obtain feature columns for explainability.

    Args:
        pipeline: Deserialized pipeline object.

    Raises:
        TypeError: If `pipeline` does not have `named_steps`.
        ValueError: If the final step is not named `'model'` or the
            preprocessing portion is not transformable.
    """

    if not hasattr(pipeline, "named_steps"):
        msg = "Pipeline must have named_steps attribute"
        logger.error(msg)
        raise TypeError(msg)

    if list(pipeline.named_steps.keys())[-1] != "model":
        msg = "The final step in the pipeline must be named 'model'"
        logger.error(msg)
        raise ValueError(msg)

    if not hasattr(pipeline[:-1], "transform"):
        msg = "Pipeline preprocessing steps must be transformable"
        logger.error(msg)
        raise ValueError(msg)

def get_feature_names(pipeline: Pipeline, X: pd.DataFrame) -> NDArray[np.str_]:
    """Obtain feature names after applying pipeline preprocessing.

    Args:
        pipeline: The full sklearn-style pipeline whose preprocessing steps
            expose a `transform` method.
        X (pandas.DataFrame): Raw input feature dataframe used for
            transformation.

    Returns:
        NDArray[np.str_]: Array of feature name strings produced by the
            preprocessing `transform` output (expects a DataFrame with
            `columns` attribute).

    Raises:
        ValueError: If the transformed output does not expose column names.
        Exception: Propagates exceptions raised during transform.
    """

    try:
        X_transformed = pipeline[:-1].transform(X)
    except Exception:
        logger.exception("Error transforming data using the pipeline")
        raise

    if hasattr(X_transformed, "columns"):
        return X_transformed.columns.to_numpy()
    else:
        msg = "Transformed data has no column names. Feature names must be provided by the transformer."
        logger.error(msg)
        raise ValueError(msg)

def validate_lengths(feature_names: NDArray[np.str_], importances: NDArray[np.float_]) -> None:
    """Ensure feature names and importance vectors have matching length.

    Args:
        feature_names (NDArray[np.str_]): Iterable of feature name strings.
        importances (NDArray[np.float_]): Iterable of numeric importances for each
            feature.

    Raises:
        ValueError: If the lengths do not match.
    """

    if len(feature_names) != len(importances):
        msg = f"Mismatch between feature names and importances: {len(feature_names)} vs {len(importances)}"
        logger.error(msg)
        raise ValueError(msg)

def get_feature_importances(feature_names: NDArray[np.str_], model: CatBoost, model_configs: dict) -> pd.DataFrame:
    """Compute and return top feature importances from a CatBoost model.

    Args:
        feature_names (NDArray[np.str_]): Feature names after preprocessing.
        model: Trained CatBoost model instance exposing
            `get_feature_importance`.
        model_configs (dict): Model configuration dict containing
            `explainability.feature_importance_method` specifying the
            CatBoost importance type.

    Returns:
        pd.DataFrame: Top 20 features with columns `feature` and
            `importance`, sorted descending by importance.

    Raises:
        KeyError: If the feature importance method is not provided in
            `model_configs`.
        Exception: If the model raises an error when computing importances.
    """

    try:
        importances = model.get_feature_importance(type=model_configs["explainability"]["feature_importance_method"])
    except KeyError:
        logger.exception("Feature importance method either not specified or invalid")
        raise
    except Exception:
        logger.exception("Error getting feature importances from the model")
        raise

    importances = np.asarray(importances)
    validate_lengths(feature_names, importances)

    df_imp_top_20 = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(20)

    return df_imp_top_20

def get_test_data(model_configs: dict) -> pd.DataFrame:
    """Load test features used for explainability computations.

    Args:
        model_configs (dict): Model configuration containing
            `features.path`, the directory where `X_test.parquet` is stored.

    Returns:
        pd.DataFrame: Test feature dataframe loaded from parquet.
    """

    features_path = Path(model_configs["features"]["path"])
    X_test = pd.read_parquet(features_path / "X_test.parquet")
    return X_test

def get_shap_importances(feature_names: NDArray[np.str_], model: CatBoost, pipeline: Pipeline, X_test: pd.DataFrame, model_configs: dict) -> pd.DataFrame:
    """Compute global SHAP importances for a CatBoost model.

    The function transforms `X_test` with the pipeline preprocessing steps,
    samples up to 1000 rows for performance, and uses a TreeExplainer to
    compute mean absolute SHAP values per feature. Currently only the
    `tree_explainer_mean_abs` method is supported.

    Args:
        feature_names (NDArray[np.str_]): Preprocessed feature names.
        model: Trained CatBoost model object.
        pipeline: Full pipeline used for preprocessing and model inference.
        X_test (pd.DataFrame): Raw test features used for transformation.
        model_configs (dict): Model configuration containing
            `explainability.shap_method`.

    Returns:
        pd.DataFrame: Top 20 features with columns `feature` and
            `mean_abs_shap`, sorted descending by mean absolute SHAP value.

    Raises:
        TypeError: If the transformed data is not a pandas DataFrame.
        ValueError: If an unsupported SHAP method is requested.
        Exception: For errors raised by SHAP computation.
    """

    X_test_transformed = pipeline[:-1].transform(X_test)

    n = min(1000, X_test_transformed.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_test_transformed.shape[0], size=n, replace=False)

    if not hasattr(X_test_transformed, "iloc"):
        raise TypeError("Transformed data must be a pandas DataFrame for SHAP analysis")
    X_test_sample = X_test_transformed.iloc[idx]

    if model_configs["explainability"]["shap_method"] == "tree_explainer_mean_abs":
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

        try:
            shap_values = explainer.shap_values(X_test_sample)
        except Exception:
            logger.exception("Error calculating SHAP values")
            raise

        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

        importances = shap_values.mean(axis=0)

        validate_lengths(feature_names, importances)

        top_20_shap_importances = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': importances,
        }).sort_values(by='mean_abs_shap', ascending=False).head(20)

        return top_20_shap_importances
    else:
        msg = f"Unsupported SHAP method: {model_configs['explainability']['shap_method']}"
        logger.error(msg)
        raise ValueError(msg)

def save_importances(df: pd.DataFrame, name_and_version: str, file_name: str, importance_type: str) -> None:
    """Persist a DataFrame of importances to CSV under the explainability folder.

    Args:
        df (pd.DataFrame): DataFrame containing importance metrics.
        name_and_version (str): Model identifier, used to create the output
            directory under `ml/models/explainability/`.
        file_name (str): CSV filename to write (e.g. `feature_importances.csv`).
        importance_type (str): Human-readable type used for logging.

    Raises:
        Exception: Propagates IO errors encountered while writing the file.
    """

    path = Path(f"ml/models/explainability/{name_and_version}")
    path.mkdir(parents=True, exist_ok=True)

    if (path / file_name).exists():
        logger.warning(f"{importance_type} importances file for model {name_and_version} already exists and will be overwritten.")

    try:
        df.to_csv(path / file_name, index=False)
    except Exception:
        logger.exception(f"Error saving {importance_type} importances of model {name_and_version}.")
        raise

    logger.info(f"{importance_type} importances of model {name_and_version} saved successfully.")

def explain_catboost(model_configs: dict) -> None:
    """Run the full explainability workflow for a CatBoost model.

    The function validates the configuration, loads the pipeline and model,
    computes the top feature importances and SHAP-based global importances,
    and writes the results to CSV files under
    `ml/models/explainability/<name>_<version>/`.

    Args:
        model_configs (dict): Model configuration dictionary containing at
            least `name`, `version`, `features.path` and `explainability`
            subsections.

    Raises:
        KeyError, ValueError, TypeError, Exception: Propagates errors thrown
            by the helper functions when configuration, pipeline, or model
            expectations are not met.
    """

    # Step 1: Validate configuration
    name_and_version = f"{model_configs['name']}_{model_configs['version']}"
    check_key_presence(model_configs)

    # Step 2: Load pipeline and model
    model, pipeline = get_pipeline_and_model(name_and_version)

    # Step 3: Inspect pipeline structure
    inspect_pipeline(pipeline)

    # Step 4: Get test data
    X_test = get_test_data(model_configs)

    # Step 5: Get feature names
    feature_names = get_feature_names(pipeline, X_test)

    # Step 6: Compute and save feature importances
    top_20_feature_importances = get_feature_importances(feature_names, model, model_configs)
    save_importances(top_20_feature_importances, name_and_version, "feature_importances.csv", "Feature")

    # Step 7: Compute and save SHAP importances
    top_20_shap_importances = get_shap_importances(feature_names, model, pipeline, X_test, model_configs)
    save_importances(top_20_shap_importances, name_and_version, "shap_importances.csv", "SHAP")