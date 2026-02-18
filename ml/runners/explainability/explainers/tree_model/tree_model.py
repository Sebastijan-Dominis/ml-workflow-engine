# TODO: Modularize and make it reusable for other tree-based models like XGBoost and LightGBM, ensuring that the code is not tightly coupled to CatBoost-specific features. This may involve creating a more generic interface for feature importance retrieval and SHAP value computation that can accommodate different model types and their respective APIs.
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from catboost import CatBoost
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import (ConfigError, DataError, ExplainabilityError,
                           PipelineContractError)
from ml.runners.explainability.classes.classes import ExplainabilityMetrics
from ml.runners.explainability.explainers.base import Explainer
from ml.utils.experiments.loading.pipeline import load_model_or_pipeline
from ml.utils.features.loading.X_and_y import load_X_and_y
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def get_feature_names_and_transformed_X(pipeline: Pipeline, X: pd.DataFrame) -> tuple[NDArray[np.str_], pd.DataFrame]:
    try:
        X_transformed = pipeline[:-1].transform(X)
    except Exception as e:
        msg = "Error transforming data using the pipeline. Ensure that the pipeline's transformers are compatible with the input data and that all necessary preprocessing steps are included."
        logger.exception(msg)
        raise PipelineContractError(msg) from e

    if hasattr(X_transformed, "columns"):
        return X_transformed.columns.to_numpy(), X_transformed
    else:
        msg = "Transformed data has no column names. Feature names must be provided by the transformer."
        logger.error(msg)
        raise DataError(msg)

def validate_lengths(feature_names: NDArray[np.str_], importances: NDArray[np.float_]) -> None:
    if len(feature_names) != len(importances):
        msg = f"Mismatch between feature names and importances: {len(feature_names)} vs {len(importances)}"
        logger.error(msg)
        raise DataError(msg)

def get_feature_importances(
    *, 
    feature_names: NDArray[np.str_], 
    pipeline: Pipeline, 
    model_cfg: TrainModelConfig, 
    top_k: int
) -> pd.DataFrame | None:
    type_param = model_cfg.explainability.methods.feature_importances.type
    model = pipeline[-1]
    try:
        if  model_cfg.explainability.methods.feature_importances.enabled:
            importances = model.get_feature_importance(type=type_param)
        else:
            msg = "Feature importance method is not enabled in the configuration. Skipping feature importance computation."
            logger.warning(msg)
            return None
        
    except AttributeError:
        msg = f"The final estimator in the pipeline does not have 'get_feature_importance' attribute. It is of type {type(model).__name__}. Ensure that the model supports feature importance computation."
        logger.error(msg)
        raise PipelineContractError(msg)
    except Exception:
        msg = f"Error retrieving feature importances from the model, using {type_param} method on a {type(model).__name__} model. Ensure that the model is properly trained and that the pipeline is correctly defined."
        logger.exception(msg)
        raise ExplainabilityError(msg)

    importances = np.asarray(importances)
    validate_lengths(feature_names, importances)

    df_imp_top_k = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(top_k)

    return df_imp_top_k

def get_shap_importances(
    *, 
    feature_names: NDArray[np.str_], 
    pipeline: Pipeline, 
    model_configs: TrainModelConfig, 
    top_k: int, 
    X_test_transformed: pd.DataFrame
    ) -> pd.DataFrame | None:
    if not model_configs.explainability.methods.shap.enabled:
        msg = "SHAP method is not enabled in the configuration. Skipping SHAP importance computation."
        logger.warning(msg)
        return None

    n = min(1000, X_test_transformed.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_test_transformed.shape[0], size=n, replace=False)

    if not hasattr(X_test_transformed, "iloc"):
        msg = "Transformed test data is not a pandas DataFrame. SHAP analysis requires the transformed data to be a DataFrame with column names."
        logger.error(msg)
        raise DataError(msg)
    
    X_test_sample = X_test_transformed.iloc[idx]

    model = pipeline[-1]
    if not isinstance(model, CatBoost):
        msg = f"TreeExplainer with mean absolute SHAP values is only supported for CatBoost models. The final estimator in the pipeline is of type {type(model).__name__}."
        logger.error(msg)
        raise PipelineContractError(msg)
    
    if model_configs.explainability.methods.shap.approximate == "tree":
        try:
            explainer = shap.TreeExplainer(
                pipeline[-1],
                feature_perturbation="tree_path_dependent",
                model_output="raw",
            )

            shap_values = explainer.shap_values(X_test_sample)
        except Exception:
            msg = "Error calculating SHAP values using TreeExplainer. Ensure that the model is compatible with Tree SHAP and that the input data is correctly preprocessed."
            logger.exception(msg)
            raise ExplainabilityError(msg)

    else:
        msg = f"Unsupported SHAP method: {model_configs.explainability.methods.shap.approximate}. Currently, only 'tree' is supported for CatBoost models."
        logger.error(msg)
        raise ConfigError(msg)

    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_values = np.abs(shap_values)

    importances = shap_values.mean(axis=0)

    validate_lengths(feature_names, importances)

    top_k_shap_importances = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': importances,
    }).sort_values(by='mean_abs_shap', ascending=False).head(top_k)

    return top_k_shap_importances

class ExplainTreeModel(Explainer):
    def explain(self, *, model_cfg: TrainModelConfig, train_dir: Path, top_k: int) -> tuple[ExplainabilityMetrics, list[dict]]:

        train_metadata_file = train_dir / "metadata.json"
        train_metadata = load_json(train_metadata_file)

        pipeline_file = Path(train_metadata.get("artifacts", {}).get("pipeline_path"))
        pipeline = load_model_or_pipeline(pipeline_file, "pipeline")

        X_test, _, feature_lineage = load_X_and_y(model_cfg, ["X_test", "y_test"], snapshot_selection=None, strict=True)

        feature_names, X_test_transformed = get_feature_names_and_transformed_X(pipeline, X_test)

        top_k_feature_importances = get_feature_importances(
            feature_names=feature_names, 
            pipeline=pipeline, 
            model_cfg=model_cfg, 
            top_k=top_k
        )

        top_k_shap_importances = get_shap_importances(
            feature_names=feature_names, 
            pipeline=pipeline, 
            model_configs=model_cfg, 
            top_k=top_k, 
            X_test_transformed=X_test_transformed
        )

        explainability_metrics = ExplainabilityMetrics(
            top_k_feature_importances=top_k_feature_importances,
            top_k_shap_importances=top_k_shap_importances,
        )

        return explainability_metrics, feature_lineage