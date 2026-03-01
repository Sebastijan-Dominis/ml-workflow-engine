import logging
from typing import Any

import numpy as np

from ml.exceptions import PipelineContractError

logger = logging.getLogger(__name__)

def ensure_1d_array(pred: Any) -> np.ndarray:
    if isinstance(pred, tuple):
        msg = "Tuple predictions are not supported. Ensure your model's predict method returns a 1D array of predictions."
        logger.error(msg)
        raise PipelineContractError(msg)

    arr = np.asarray(pred)

    if arr.ndim != 1:
        msg = f"Expected 1D array of predictions, but got array with shape {arr.shape}."
        logger.error(msg)
        raise PipelineContractError(msg)

    return arr