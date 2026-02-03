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
import yaml
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier, CatBoostRegressor

# Project imports
from ml.training.train_scripts.utils import load_train_and_val_data
from ml.utils.features import load_schemas, get_cat_features
from ml.pipelines.builders import build_pipeline
from ml.registry.model_classes import MODEL_CLASS_REGISTRY

def prepare_model(cfg_model_specs, cfg_train, cat_features):
    model = MODEL_CLASS_REGISTRY[cfg_model_specs["model_class"]](
        # Basic hyperparameters
        **cfg_train["train_params"],
        iterations=cfg_train["train_params"].get("iterations", 1), # Higher default for final training
        task_type='CPU',             # CPU for training for compatibility
        verbose=100,                
        random_state=cfg_model_specs['seed'],
        cat_features=cat_features,
        early_stopping_rounds=100,
        class_weights=cfg_model_specs.get("class_weights", None)
    )
    return model

def train_catboost_model(
    steps: list,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> tuple[CatBoostClassifier | CatBoostRegressor, Pipeline]:
    """Fit preprocessing steps and the CatBoost model, returning a Pipeline.

    The function separates preprocessing steps from the final model, fits
    the preprocessing pipeline on the training data, transforms both train
    and validation sets, and fits the CatBoost model using the transformed
    data and validation set for early stopping.

    Args:
        steps (list): Pipeline steps where the last element is the model.
        X_train, y_train, X_val, y_val: Training and validation data.

    Returns:
        tuple[CatBoostClassifier | CatBoostRegressor, Pipeline]: A fitted ``sklearn.pipeline.Pipeline`` combining preprocessing
        and the trained model.
    """

    # Separate preprocessing and model without using the steps variable
    preprocessing_pipeline = Pipeline(steps[:-1])
    model = steps[-1][1]

    X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_val_processed = preprocessing_pipeline.transform(X_val)

    model.fit(
        X_train_processed,
        y_train,
        eval_set=(X_val_processed, y_val),
        use_best_model=True,
    )

    pipeline = Pipeline(steps=steps[:-1] + [("model", model)])

    return model, pipeline

def train_catboost(cfg_model_specs: dict, cfg_train: dict) -> tuple[CatBoostClassifier | CatBoostRegressor, Pipeline]:
    """Train a binary classification model using CatBoost and project components.

    This is the high-level routine used by the training CLI to execute a
    complete training run. It performs data loading, dynamic component
    import for model-specific preprocessing, model construction, pipeline
    assembly, training, and returns a fitted pipeline object.

    Args:
        name_version (str): Component module name under ``ml.components``.
        cfg (dict): Validated configuration dictionary.

    Returns:
        tuple[CatBoostClassifier | CatBoostRegressor, Pipeline]: A fitted sklearn Pipeline containing preprocessing and
        the trained CatBoost model.

    Raises:
        Exception: Any exception during the training process is logged and
        re-raised to signal a fatal training error.
    """

    try:
        X_train, y_train, X_val, y_val = load_train_and_val_data(cfg_model_specs) # Load data

        features_path = Path(cfg_model_specs["features"]["path"])
        raw_schema, derived_schema = load_schemas(features_path, logger)

        cat_features = get_cat_features(raw_schema, derived_schema)

        model_class = prepare_model(cfg_model_specs, cfg_train, cat_features) # Define model

        if not isinstance(model_class, CatBoostClassifier or CatBoostRegressor):
            logger.error("Defined model is not a CatBoostClassifier or CatBoostRegressor instance.")
            raise TypeError("Defined model is not a CatBoostClassifier or CatBoostRegressor instance.")

        pipeline = build_pipeline(
            pipeline_cfg=yaml.safe_load(
                open(f"configs/pipelines/{cfg_model_specs['pipeline']}.yaml")
            ),
            raw_schema=raw_schema,
            derived_schema=derived_schema,
        ) # Build preprocessing pipeline

        pipeline.steps.append(("Model", model_class)) # Append model to pipeline

        model_trained, pipeline_trained = train_catboost_model(pipeline.steps, X_train, y_train, X_val, y_val) # Train model

        logger.info(f"Model {cfg_model_specs['problem']}_{cfg_model_specs['segment']['name']}_{cfg_model_specs['version']} trained successfully.") # Log success

        return model_trained, pipeline_trained # Return fitted pipeline
    except Exception:
        logger.exception(f"Training failed for model {cfg_model_specs['problem']}_{cfg_model_specs['segment']['name']}_{cfg_model_specs['version']}") # Log failure
        raise