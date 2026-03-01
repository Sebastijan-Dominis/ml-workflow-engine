import logging
from typing import Any

from catboost import CatBoostClassifier, CatBoostRegressor

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import UserError
from ml.registry.model_classes import MODEL_CLASS_REGISTRY
from ml.registry.regression_loss_functions import REGRESSION_LOSS_FUNCTIONS

logger = logging.getLogger(__name__)

def extract_catboost_params(model_cfg: TrainModelConfig) -> dict[str, Any]:
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
    class_weights: dict
) -> CatBoostClassifier | CatBoostRegressor:

    best_params = extract_catboost_params(model_cfg)

    model_kwargs = dict(
        iterations=model_cfg.training.iterations,
        task_type=model_cfg.training.hardware.task_type.value,
        verbose=model_cfg.verbose,
        random_state=model_cfg.seed,
        cat_features=cat_features,
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
            logger.debug(f"Using default regression metric RMSE.")
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

    logger.info(
        f"Hardware settings for training of {model_cfg.problem} {model_cfg.segment.name} {model_cfg.version} | "
        f"Task type: {model_cfg.training.hardware.task_type.value}, "
        f"Devices: {model_cfg.training.hardware.devices}"
    )

    return model