"""Schema validation transformer for model input integrity checks."""

import logging

from ml.exceptions import DataError

from ..base import PipelineComponent

logger = logging.getLogger(__name__)

class SchemaValidator(PipelineComponent):
    """Validate that incoming DataFrame contains required columns.

    Args:
        required_features (list): List of column names expected in the input.
    """

    def __init__(self, required_features):
        """Initialize validator with required input feature names.

        Args:
            required_features: Required feature names.

        Returns:
            None: Initializes validator state.
        """
        self.required_features = required_features

    def transform(self, X):
        """Validate required columns are present and return input unchanged.

        Args:
            X: Input dataframe.

        Returns:
            pd.DataFrame: Original dataframe when schema validation passes.
        """
        missing = [c for c in self.required_features if c not in X.columns]
        if missing:
            msg = f"Model input schema violation. Missing columns: {missing}"
            logger.error(msg)
            raise DataError(msg)
        return X
