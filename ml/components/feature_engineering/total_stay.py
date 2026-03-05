"""Feature operator for computing total stay duration in nights."""

from ml.components.base import SklearnFeatureMixin
from ml.components.feature_engineering.base import FeatureOperator


class TotalStay(FeatureOperator, SklearnFeatureMixin):
    """Create ``total_stay`` as weekend plus weekday nights."""

    output_features = ["total_stay"]

    def transform(self, X):
        """Compute total number of booked nights for each reservation.

        Args:
            X: Input feature frame containing weekday and weekend night counts.

        Returns:
            DataFrame with an added ``total_stay`` feature.
        """
        if not hasattr(self, "n_features_in_"):
            SklearnFeatureMixin.fit(self, X)

        X = X.copy()
        X["total_stay"] = (
            X["stays_in_weekend_nights"] + X["stays_in_week_nights"]
        )
        return X
