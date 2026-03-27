"""Module for prediction execution logic."""
import logging
from typing import Any

import pandas as pd

from ml.exceptions import InferenceError

logger = logging.getLogger(__name__)

def predict(X: pd.DataFrame, artifact: Any) -> tuple[pd.Series, pd.DataFrame]:
    """Generate predictions and probabilities using the provided artifact.

    Args:
        X: Input features for prediction.
        artifact: The trained model artifact.

    Returns:
        A tuple containing the predictions and probabilities.
    """
    try:
        preds = pd.Series(artifact.predict(X), index=X.index)

        if hasattr(artifact, "predict_proba"):
            proba = pd.DataFrame(artifact.predict_proba(X), index=X.index)
        else:
            proba = pd.DataFrame()
    except Exception as e:
        msg = "Error during prediction. "
        logger.exception(msg)
        raise InferenceError(msg) from e

    return preds, proba
