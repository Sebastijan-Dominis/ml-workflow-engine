import logging

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline
from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig

from ml.exceptions import PipelineContractError
from ml.pipelines.builders import build_pipeline
from ml.utils.add_model_to_pipeline import add_model_to_pipeline

logger = logging.getLogger(__name__)

def build_pipeline_with_model(
    *,
    model_cfg: SearchModelConfig | TrainModelConfig,
    pipeline_cfg: dict, 
    input_schema: pd.DataFrame, 
    derived_schema: pd.DataFrame,
    model: CatBoostClassifier | CatBoostRegressor
) -> Pipeline:
    pipeline = build_pipeline(
        model_cfg=model_cfg,
        pipeline_cfg=pipeline_cfg,
        input_schema=input_schema,
        derived_schema=derived_schema
    )

    if not isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
        msg = "Defined model is not a CatBoostClassifier or CatBoostRegressor instance."
        logger.error(msg)
        raise PipelineContractError(msg)

    pipeline = add_model_to_pipeline(pipeline, model)
    logger.debug("Model added to pipeline successfully.")
    return pipeline