"""Binary classification training utilities using CatBoost.

This module contains helper routines and a top-level training function
used to train binary classification models with CatBoost within the
project training framework. It provides deterministic data loading,
dynamic import of model-specific pipeline components from the
``ml.components`` package, pipeline construction, model definition,
and training orchestration.

The public entrypoint is ``train_binary_classification_with_catboost`` which
returns a fitted ``sklearn.pipeline.Pipeline`` combining preprocessing
steps and the trained CatBoost model.
"""

# General imports
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import importlib
from typing import Optional, List

from pathlib import Path
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

# Utility imports
from ml.training.train_scripts.utils import load_train_and_val_data, import_components, get_features_from_schema, define_catboost_model, train_catboost_model

def build_pipeline_steps(
    cfg: dict,
    SchemaValidator: type,
    FillCategoricalMissing: type,
    FeatureEngineer: type,
    FeatureSelector: type,
    model_class: CatBoostClassifier,
    required_features: list,
    categorical_features: list,
    created_columns: Optional[List[str]] = None,
) -> list:
    """Assemble a list of (name, transformer) steps for a sklearn Pipeline.

    The function inspects pipeline flags in ``cfg['pipeline']`` and
    conditionally includes validators, missing-value fillers, feature
    engineering, and feature selection steps. The final step is the
    CatBoost model instance.

    Args:
        cfg (dict): Configuration controlling which pipeline steps to include.
        SchemaValidator, FillCategoricalMissing, FeatureEngineer, FeatureSelector:
            Classes or callables used to construct pipeline transformers.
        model_class (CatBoostClassifier): The model instance to attach.
        required_features (list): Columns required for schema validation.
        categorical_features (list): Columns treated as categorical.
        created_columns (list, optional): Columns created by feature engineering.
    Returns:
        list: Pipeline steps as expected by ``sklearn.pipeline.Pipeline``.
    """

    steps = []
    if cfg["pipeline"]["validate_schema"]:
        steps.append(("schema_validator", SchemaValidator(required_columns=required_features)))
    if cfg["pipeline"]["fill_categorical_missing"]:
        steps.append(("fill_categorical_missing", FillCategoricalMissing(categorical_columns=categorical_features)))
    if cfg["pipeline"]["feature_engineering"]:
        steps.append(("feature_engineering", FeatureEngineer(created_columns=created_columns)))
    if cfg["pipeline"]["feature_selection"]:
        # Make sure FeatureEngineer contains created_columns attribute
        steps.append(("feature_selector", FeatureSelector(selected_columns=required_features + (created_columns or []))))

    steps.append(("model", model_class))

    return steps



def train_binary_classification_with_catboost(name_version: str, cfg: dict) -> Pipeline:
    """Train a binary classification model using CatBoost and project components.

    This is the high-level routine used by the training CLI to execute a
    complete training run. It performs data loading, dynamic component
    import for model-specific preprocessing, model construction, pipeline
    assembly, training, and returns a fitted pipeline object.

    Args:
        name_version (str): Component module name under ``ml.components``.
        cfg (dict): Validated configuration dictionary.

    Returns:
        Pipeline: A fitted sklearn Pipeline containing preprocessing and
        the trained CatBoost model.

    Raises:
        Exception: Any exception during the training process is logged and
        re-raised to signal a fatal training error.
    """

    try:
        X_train, y_train, X_val, y_val = load_train_and_val_data(cfg) # Load data

        (
            SchemaValidator,
            FillCategoricalMissing,
            FeatureEngineer,
            FeatureSelector,
        ) = import_components(cfg["artifacts"]["components_path"], cfg["pipeline"]) # Import components

        categorical_features, required_features, cat_features = get_features_from_schema(cfg, cfg["features"]["schema_path"])
        
        model_class = define_catboost_model(cfg, cat_features) # Define model

        if not isinstance(model_class, CatBoostClassifier):
            logger.error("Defined model is not a CatBoostClassifier instance.")
            raise TypeError("Defined model is not a CatBoostClassifier instance.")

        steps = build_pipeline_steps(
            cfg,
            SchemaValidator,
            FillCategoricalMissing,
            FeatureEngineer,
            FeatureSelector,
            model_class,
            required_features,
            categorical_features,
            cfg.get("created_columns", None)
        )

        pipeline = train_catboost_model(steps, X_train, y_train, X_val, y_val) # Train model

        logger.info(f"Model {cfg['name']}_{cfg['version']} trained successfully.") # Log success

        return pipeline # Return fitted pipeline
    except Exception:
        logger.exception(f"Training failed for model {cfg['name']}_{cfg['version']}") # Log failure
        raise