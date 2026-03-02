"""CatBoost estimator preparation utilities for search phases."""

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
    class_weights: dict | None
) -> CatBoostClassifier | CatBoostRegressor:
    """Build CatBoost model configured for broad or narrow search phase.

    Args:
        model_cfg: Validated search model configuration.
        search_phase: Search phase whose settings should be applied.
        cat_features: Categorical feature indices or names for CatBoost.
        class_weights: Optional class-weight mapping for classification tasks.

    Returns:
        Configured CatBoost classifier or regressor instance.
    """

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

    logger.info("Model set to use task_type: %s, devices: %s", model.get_params().get("task_type"), model.get_params().get("devices", "N/A"))

    return model