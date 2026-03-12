"""Low-level CatBoost fitting helper used by training runner."""

import logging

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import TrainingError
from ml.pipelines.composition.add_model_to_pipeline import add_model_to_pipeline

logger = logging.getLogger(__name__)

def train_catboost_model(
    model_cfg: TrainModelConfig,
    *,
    steps: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> tuple[CatBoostClassifier | CatBoostRegressor, Pipeline]:
    """Fit preprocessing steps and CatBoost model, returning trained pipeline.

    The function separates preprocessing steps from the final model, fits
    the preprocessing pipeline on the training data, transforms both train
    and validation sets, and fits the CatBoost model using the transformed
    data and validation set for early stopping.

    Args:
        model_cfg (TrainModelConfig): Configuration object for the model.
        steps (list): Pipeline steps where the last element is the model.
        X_train, y_train, X_val, y_val: Training and validation data.

    Returns:
        tuple[CatBoostClassifier | CatBoostRegressor, Pipeline]: Trained model
        and fitted preprocessing+model pipeline.
    """
    try:
        # Separate preprocessing and model without using the steps variable
        preprocessing_pipeline = Pipeline(steps[:-1])
        model = steps[-1][1]

        X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train)

        fit_kwargs: dict = {
            "use_best_model": True
        }

        fit_kwargs.update({
            "save_snapshot": True,
            "snapshot_file": "catboost_snapshot.bin",
            "snapshot_interval": model_cfg.training.snapshot_interval_seconds,
        })

        if model_cfg.training.early_stopping_rounds:
            X_val_processed = preprocessing_pipeline.transform(X_val)
            fit_kwargs["eval_set"] = (X_val_processed, y_val)
            fit_kwargs["early_stopping_rounds"] = model_cfg.training.early_stopping_rounds

        logger.info("Training the model...")

        model.fit(
            X_train_processed,
            y_train,
            **fit_kwargs
        )

        logger.info("Model trained successfully.")

        pipeline = add_model_to_pipeline(preprocessing_pipeline, model)

        return model, pipeline
    except Exception as e:
        msg = f"Error during training of {model_cfg.problem}_{model_cfg.segment.name}_{model_cfg.version} with CatBoost: {str(e)}"
        logger.error(msg)
        raise TrainingError(msg) from e
