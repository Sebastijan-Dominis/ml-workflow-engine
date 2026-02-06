import logging
logger = logging.getLogger(__name__)
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline
import pandas as pd

from ml.pipelines.builders import build_pipeline
from ml.registry.model_classes import MODEL_CLASS_REGISTRY
from ml.exceptions import PipelineContractError
from ml.config.validation_schemas.model_cfg import SearchModelConfig

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

def add_model_to_pipeline(pipeline: Pipeline, model: CatBoostClassifier | CatBoostRegressor) -> Pipeline:
    pipeline_with_model = Pipeline(pipeline.steps + [("Model", model)])
    return pipeline_with_model

def build_pipeline_with_model(pipeline_cfg: dict, input_schema: pd.DataFrame, derived_schema: pd.DataFrame, model: CatBoostClassifier | CatBoostRegressor) -> Pipeline:
    pipeline = build_pipeline(pipeline_cfg, input_schema, derived_schema)

    if not isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
        msg = "Defined model is not a CatBoostClassifier or CatBoostRegressor instance."
        logger.error(msg)
        raise PipelineContractError(msg)

    pipeline = add_model_to_pipeline(pipeline, model)
    logger.debug("Model added to pipeline successfully.")
    return pipeline