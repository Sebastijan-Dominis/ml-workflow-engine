import logging

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import TrainingError

logger = logging.getLogger(__name__)

def train_catboost_model(
    model_cfg: TrainModelConfig,
    *,
    steps: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[CatBoostClassifier | CatBoostRegressor, Pipeline]:
    """Fit preprocessing steps and the CatBoost model, returning a Pipeline.

    The function separates preprocessing steps from the final model, fits
    the preprocessing pipeline on the training data, transforms both train
    and validation sets, and fits the CatBoost model using the transformed
    data and validation set for early stopping.

    Args:
        model_cfg (TrainModelConfig): Configuration object for the model.
        steps (list): Pipeline steps where the last element is the model.
        X_train, y_train, X_val, y_val: Training and validation data.

    Returns:
        tuple[CatBoostClassifier | CatBoostRegressor, Pipeline]: A fitted ``sklearn.pipeline.Pipeline`` combining preprocessing
        and the trained model.
    """
    try:
        # Separate preprocessing and model without using the steps variable
        preprocessing_pipeline = Pipeline(steps[:-1])
        model = steps[-1][1]

        X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train)

        fit_kwargs: dict = {
            "use_best_model": True
        }

        if model_cfg.training.early_stopping_rounds:
            X_val_processed = preprocessing_pipeline.transform(X_val)
            fit_kwargs["eval_set"] = (X_val_processed, y_val)
            fit_kwargs["early_stopping_rounds"] = model_cfg.training.early_stopping_rounds

        model.fit(
            X_train_processed,
            y_train,
            **fit_kwargs
        )

        pipeline = Pipeline(steps=steps[:-1] + [("model", model)])

        logger.info(f"Model {model_cfg.problem}_{model_cfg.segment.name}_{model_cfg.version} trained successfully.")

        return model, pipeline
    except Exception as e:
        msg = f"Error during training of {model_cfg.problem}_{model_cfg.segment.name}_{model_cfg.version} with CatBoost: {str(e)}"
        logger.error(msg)
        raise TrainingError(msg) from e