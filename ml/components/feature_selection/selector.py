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
        self.selected_features = selected_features

    def transform(self, X):
        # select only required columns, ignore extras
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            msg = f"Missing columns: {missing}"
            logger.error(msg)
            raise DataError(msg)
        return X[self.selected_features]