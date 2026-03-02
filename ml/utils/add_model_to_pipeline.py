"""Helpers for appending validated model instances to sklearn pipelines."""

import logging
from typing import Any

from sklearn.pipeline import Pipeline

from ml.exceptions import PipelineContractError
from ml.registry.model_classes import MODEL_CLASS_REGISTRY

logger = logging.getLogger(__name__)

def add_model_to_pipeline(pipeline: Pipeline, model: Any) -> Pipeline:
    """Append a supported estimator as the final `Model` step in a pipeline.

    Args:
        pipeline: Existing sklearn pipeline.
        model: Estimator instance expected to be supported by registry.

    Returns:
        Pipeline: New pipeline instance with appended `Model` step.
    """

    if not isinstance(model, tuple(MODEL_CLASS_REGISTRY.values())):
        msg = f"The provided model is not supported. Expected one of: {list(MODEL_CLASS_REGISTRY.values())}, but got {type(model)}."
        logger.error(msg)
        raise PipelineContractError(msg)
    
    pipeline_with_model = Pipeline(pipeline.steps + [("Model", model)])
    logger.debug("Model added to pipeline successfully.")
    return pipeline_with_model
