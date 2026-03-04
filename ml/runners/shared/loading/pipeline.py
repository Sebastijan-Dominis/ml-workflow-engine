"""Utilities for loading serialized model and pipeline artifacts safely."""

import logging
from pathlib import Path
from typing import Literal, overload

import joblib
from ml.exceptions import PipelineContractError
from ml.types import AllowedModels
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

@overload
def load_model_or_pipeline(file: Path, target_type: Literal["model"]) -> AllowedModels:
    """Typed overload for loading serialized estimator/model artifacts.

    Args:
        file: Serialized artifact file path.
        target_type: Artifact type discriminator.

    Returns:
        AllowedModels: Deserialized model instance.
    """
    ...

@overload
def load_model_or_pipeline(file: Path, target_type: Literal["pipeline"]) -> Pipeline:
    """Typed overload for loading serialized sklearn pipeline artifacts.

    Args:
        file: Serialized artifact file path.
        target_type: Artifact type discriminator.

    Returns:
        Pipeline: Deserialized sklearn pipeline instance.
    """
    ...

def load_model_or_pipeline(file: Path, target_type: Literal["model", "pipeline"]) -> AllowedModels | Pipeline:
    """Load and type-validate a serialized model or pipeline artifact from disk.

    Args:
        file: Serialized artifact file path.
        target_type: Artifact type discriminator.

    Returns:
        AllowedModels | Pipeline: Deserialized artifact with expected type.
    """

    if not file.exists():
        msg = f"File not found at {file}"
        logger.error(msg)
        raise PipelineContractError(msg) from None

    try:
        with open(file, "rb") as f:
            output = joblib.load(f)
    except Exception as e:
        msg = f"Error loading from {file}."
        logger.exception(msg)
        raise PipelineContractError(msg) from e

    if target_type == "pipeline":
        if not isinstance(output, Pipeline):
            msg = f"Expected a Pipeline object when loading a pipeline. Got type {type(output)} instead."
            logger.error(msg)
            raise PipelineContractError(msg) from None
        return output
    elif target_type == "model":
        if not isinstance(output, AllowedModels):
            msg = f"Expected a model object of type {AllowedModels} when loading a model. Got type {type(output)} instead."
            logger.error(msg)
            raise PipelineContractError(msg) from None
        return output
