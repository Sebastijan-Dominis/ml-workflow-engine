"""Validation helpers for explainability vector/feature alignment checks."""

import logging

import numpy as np
from ml.exceptions import DataError
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

def validate_lengths(feature_names: NDArray[np.str_], importances: NDArray[np.float64]) -> None:
    """Ensure feature-name and importance arrays have identical lengths.

    Args:
        feature_names: Feature-name array.
        importances: Importance-value array.

    Returns:
        None.
    """

    if len(feature_names) != len(importances):
        msg = f"Mismatch between feature names and importances: {len(feature_names)} vs {len(importances)}"
        logger.error(msg)
        raise DataError(msg)
