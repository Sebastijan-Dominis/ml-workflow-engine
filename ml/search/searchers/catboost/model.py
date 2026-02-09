import logging

from catboost import CatBoostClassifier, CatBoostRegressor

from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.registry.model_classes import MODEL_CLASS_REGISTRY

logger = logging.getLogger(__name__)

def prepare_model(model_cfg: SearchModelConfig, search_phase: str, cat_features: list) -> CatBoostClassifier | CatBoostRegressor:
    search_phase_cfg = getattr(model_cfg.search, search_phase)
    model = MODEL_CLASS_REGISTRY[model_cfg.model_class](
        # Basic hyperparameters
        iterations=search_phase_cfg.iterations,
        task_type=model_cfg.search.hardware.task_type.value,           
        devices=model_cfg.search.hardware.devices,                
        verbose=model_cfg.verbose,               
        random_state=model_cfg.seed,
        cat_features=cat_features,
        class_weights=getattr(model_cfg, "class_weights", None)
    )

    logger.info(f"Hardware settings for {search_phase} search of {model_cfg.problem} {model_cfg.segment.name} {model_cfg.version} | Task type: {model_cfg.search.hardware.task_type.value}, Devices: {model_cfg.search.hardware.devices}")

    return model
