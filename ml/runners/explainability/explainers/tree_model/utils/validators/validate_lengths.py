import logging

import numpy as np
from numpy.typing import NDArray

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def validate_lengths(feature_names: NDArray[np.str_], importances: NDArray[np.float_]) -> None:
    if len(feature_names) != len(importances):
        msg = f"Mismatch between feature names and importances: {len(feature_names)} vs {len(importances)}"
        logger.error(msg)
        raise DataError(msg)