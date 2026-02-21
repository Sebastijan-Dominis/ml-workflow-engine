import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from ml.exceptions import DataError, PipelineContractError

logger = logging.getLogger(__name__)
def get_feature_names_and_transformed_X(pipeline: Pipeline, X: pd.DataFrame) -> tuple[NDArray[np.str_], pd.DataFrame]:
    try:
        X_transformed = pipeline[:-1].transform(X)
    except Exception as e:
        msg = "Error transforming data using the pipeline. Ensure that the pipeline's transformers are compatible with the input data and that all necessary preprocessing steps are included."
        logger.exception(msg)
        raise PipelineContractError(msg) from e

    if hasattr(X_transformed, "columns"):
        return X_transformed.columns.to_numpy(), X_transformed
    else:
        msg = "Transformed data has no column names. Feature names must be provided by the transformer."
        logger.error(msg)
        raise DataError(msg)