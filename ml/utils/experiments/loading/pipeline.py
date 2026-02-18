import logging
from pathlib import Path
from typing import Literal, overload

import joblib
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from ml.exceptions import PipelineContractError

# Update this type alias if additional model types are added in the future
AllowedModels = CatBoostClassifier

logger = logging.getLogger(__name__)

@overload
def load_model_or_pipeline(file: Path, target_type: Literal["model"]) -> AllowedModels: ...

@overload
def load_model_or_pipeline(file: Path, target_type: Literal["pipeline"]) -> Pipeline: ...

def load_model_or_pipeline(file: Path, target_type: Literal["model", "pipeline"]) -> AllowedModels | Pipeline:
    """Load a serialized pipeline from disk.

    Args:
        file (pathlib.Path): Path to the serialized pipeline file.

    Returns:
        AllowedModels | Pipeline: Deserialized model/estimator.
    """
    if not file.exists():
        msg = f"File not found at {file}"
        logger.error(msg)
        raise PipelineContractError(msg)
    
    try:
        with open(file, "rb") as f:
            output = joblib.load(f)
    except Exception as e:
        msg = f"Error loading from {file}: {e}"
        logger.error(msg)
        raise PipelineContractError(msg)
    
    if target_type == "pipeline":
        if not isinstance(output, Pipeline):
            msg = f"Expected a Pipeline object when loading a pipeline. Got type {type(output)} instead."
            logger.error(msg)
            raise PipelineContractError(msg)
        return output
    elif target_type == "model":
        if not isinstance(output, AllowedModels):
            msg = f"Expected a model object of type {AllowedModels} when loading a model. Got type {type(output)} instead."
            logger.error(msg)
            raise PipelineContractError(msg)
        return output