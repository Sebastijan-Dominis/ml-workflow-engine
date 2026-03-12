"""CatBoost-specific model parameter extraction and estimator preparation."""

import logging
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier, CatBoostRegressor

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import UserError
from ml.registries.catalogs import MODEL_CLASS_REGISTRY, REGRESSION_LOSS_FUNCTIONS

logger = logging.getLogger(__name__)

def extract_catboost_params(model_cfg: TrainModelConfig) -> dict[str, Any]:
    """Extract non-null CatBoost model and ensemble params from config.

    Args:
        model_cfg: Training model configuration.

    Returns:
        dict[str, Any]: Flattened CatBoost parameter mapping.
    """

    params: dict[str, Any] = {}

    # model params
    if model_cfg.training.model:
        params.update(
            {
                k: v
                for k, v in model_cfg.training.model.model_dump(exclude_none=True).items()
            }
        )

    # ensemble params
    if model_cfg.training.ensemble:
        params.update(
            {
                k: v
                for k, v in model_cfg.training.ensemble.model_dump(exclude_none=True).items()
            }
        )

    return params

def prepare_model(
    model_cfg: TrainModelConfig,
    *,
    cat_features: list,
    class_weights: dict,
    failure_management_dir: Path
) -> CatBoostClassifier | CatBoostRegressor:
    """Construct configured CatBoost estimator for the current training run.

    Args:
        model_cfg: Training model configuration.
        cat_features: Categorical feature names.
        class_weights: Class-weight payload.
        failure_management_dir: Directory used by CatBoost for logs/artifacts.

    Returns:
        CatBoostClassifier | CatBoostRegressor: Configured CatBoost estimator instance.
    """

    best_params = extract_catboost_params(model_cfg)

    model_kwargs = dict(
        iterations=model_cfg.training.iterations,
        task_type=model_cfg.training.hardware.task_type.value,
        verbose=model_cfg.verbose,
        random_state=model_cfg.seed,
        cat_features=cat_features,
        train_dir=str(failure_management_dir),  # CatBoost expects a string path for train_dir
        **best_params
    )

    # Add GPU devices if task type is GPU
    if model_cfg.training.hardware.task_type.value.lower() == "gpu":
        model_kwargs["devices"] = model_cfg.training.hardware.devices

    # Only for classifier and weighting enabled
    if (
        model_cfg.task.type == "classification"
        and model_cfg.class_weighting.policy != "off"
        and class_weights
    ):
        model_kwargs["class_weights"] = class_weights.get("class_weights")

    if model_cfg.task.type == "regression":
        loss_function = None
        if model_cfg.scoring.policy == "regression_default":
            loss_function = "RMSE"
            logger.debug("Using default regression metric RMSE.")
        elif model_cfg.scoring.policy == "fixed":
            if model_cfg.scoring.fixed_metric in REGRESSION_LOSS_FUNCTIONS:
                loss_function = REGRESSION_LOSS_FUNCTIONS[model_cfg.scoring.fixed_metric]
                logger.debug(f"Using fixed regression metric {model_cfg.scoring.fixed_metric} mapped to CatBoost loss function {loss_function}")
            else:
                msg = f"Unsupported fixed metric {model_cfg.scoring.fixed_metric} for regression task."
                logger.error(msg)
                raise UserError(msg)
        if loss_function:
            model_kwargs["loss_function"] = loss_function

    model = MODEL_CLASS_REGISTRY[model_cfg.model_class](**model_kwargs)

    logger.info("Model set to use task_type: %s, devices: %s", model.get_params().get("task_type"), model.get_params().get("devices", "N/A"))

    return model
