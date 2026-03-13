"""Helpers for constructing CatBoost-ready sklearn pipelines."""

import logging

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline

from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import PipelineContractError
from ml.pipelines.builders import build_pipeline
from ml.pipelines.composition.add_model_to_pipeline import add_model_to_pipeline
from ml.pipelines.models import PipelineConfig

logger = logging.getLogger(__name__)

def build_pipeline_with_model(
    *,
    model_cfg: SearchModelConfig | TrainModelConfig,
    pipeline_cfg: PipelineConfig,
    input_schema: pd.DataFrame,
    derived_schema: pd.DataFrame,
    model: CatBoostClassifier | CatBoostRegressor
) -> Pipeline:
    """Build a feature pipeline and append a validated CatBoost model step.

    Args:
        model_cfg: Validated search or train model configuration.
        pipeline_cfg: Pipeline configuration object.
        input_schema: Input feature schema dataframe.
        derived_schema: Derived feature schema dataframe.
        model: Instantiated CatBoost estimator.

    Returns:
        Pipeline: Constructed sklearn pipeline with final `Model` step attached.
    """

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
    return pipeline
