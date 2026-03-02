"""Feature operator for per-person average daily rate (ADR)."""

from ml.components.base import SklearnFeatureMixin
from ml.components.feature_engineering.base import FeatureOperator


class AdrPerPerson(FeatureOperator, SklearnFeatureMixin):
    """Create ``adr_per_person`` from ADR and party-size columns."""

    output_features = ["adr_per_person"]

    def transform(self, X):
        """Compute ADR normalized by number of guests in each booking.

        Args:
            X: Input feature frame containing ``adr``, ``adults``, ``children``,
                and ``babies`` columns.

        Returns:
            DataFrame with an added ``adr_per_person`` feature.
        """
        if not hasattr(self, "n_features_in_"):
            self.fit(X)

        denom = (
            X["adults"] + X["children"] + X["babies"]
        ).replace(0, 1)
        X = X.copy()
        X["adr_per_person"] = X["adr"] / denom
        return X