import logging
from typing import Any, Dict

from catboost import CatBoostClassifier, CatBoostRegressor

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.registry.model_classes import MODEL_CLASS_REGISTRY

logger = logging.getLogger(__name__)

def extract_catboost_params(model_cfg: TrainModelConfig) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

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

    model = MODEL_CLASS_REGISTRY[model_cfg.model_class](
        # Basic hyperparameters
        iterations=model_cfg.training.iterations,
        task_type=model_cfg.training.hardware.task_type.value,           
        devices=model_cfg.training.hardware.devices,                
        verbose=model_cfg.verbose,               
        random_state=model_cfg.seed,
        cat_features=cat_features,
        class_weights=class_weights,
        **best_params
    )

    logger.info(f"Hardware settings for training of {model_cfg.problem} {model_cfg.segment.name} {model_cfg.version} | Task type: {model_cfg.training.hardware.task_type.value}, Devices: {model_cfg.training.hardware.devices}")

    return model