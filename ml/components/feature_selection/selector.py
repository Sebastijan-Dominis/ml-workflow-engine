"""Feature selection transformer for fixed-column model inputs."""

import logging

from ml.components.base import SklearnFeatureMixin
from ml.exceptions import DataError

logger = logging.getLogger(__name__)

class FeatureSelector(SklearnFeatureMixin):
    """Select a fixed set of columns for model training.

    The transformer validates presence of selected columns and returns a
    DataFrame with only the requested columns.
    """

    def __init__(self, selected_features):
        """Initialize selector with the required feature list.

        Args:
            selected_features: Feature names to keep.

        Returns:
            None: Initializes selector state.
        """
        self.selected_features = selected_features

    def transform(self, X):
        """Return only selected features after validating their presence.

        Args:
            X: Input dataframe.

        Returns:
            pd.DataFrame: Dataframe restricted to selected features.
        """
        # select only required columns, ignore extras
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            msg = f"Missing columns: {missing}"
            logger.error(msg)
            raise DataError(msg)
        return X[self.selected_features]