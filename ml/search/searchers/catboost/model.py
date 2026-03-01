import logging

from catboost import CatBoostClassifier, CatBoostRegressor
from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.registry.model_classes import MODEL_CLASS_REGISTRY
from ml.search.constants import SEARCH_PHASES

logger = logging.getLogger(__name__)

def prepare_model(
    model_cfg: SearchModelConfig, 
    *,
    search_phase: SEARCH_PHASES, 
    cat_features: list, 
    class_weights: dict
) -> CatBoostClassifier | CatBoostRegressor:

    search_phase_cfg = getattr(model_cfg.search, search_phase)

    # Base kwargs shared by classifier & regressor
    model_kwargs = dict(
        iterations=search_phase_cfg.iterations,
        task_type=model_cfg.search.hardware.task_type.value,
        devices=model_cfg.search.hardware.devices,
        verbose=model_cfg.verbose,
        random_state=model_cfg.seed,
        cat_features=cat_features,
    )

    # Add class_weights ONLY for classifier AND when weighting is enabled
    if (
        model_cfg.model_class == "classifier"
        and model_cfg.class_weighting.policy != "off"
        and class_weights
    ):
        model_kwargs["class_weights"] = class_weights.get("class_weights")

    model = MODEL_CLASS_REGISTRY[model_cfg.model_class](**model_kwargs)

    logger.info(
        f"Hardware settings for {search_phase} search of "
        f"{model_cfg.problem} {model_cfg.segment.name} {model_cfg.version} | "
        f"Task type: {model_cfg.search.hardware.task_type.value}, "
        f"Devices: {model_cfg.search.hardware.devices}"
    )

    return model