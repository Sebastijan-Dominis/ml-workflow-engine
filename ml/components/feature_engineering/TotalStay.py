from .base import FeatureOperator
from ..base import SklearnFeatureMixin

class TotalStay(FeatureOperator, SklearnFeatureMixin):
    output_features = ["total_stay"]

    def transform(self, X):
        if not hasattr(self, "n_features_in_"):
            self.fit(X)

        X = X.copy()
        X["total_stay"] = (
            X["stays_in_weekend_nights"] + X["stays_in_week_nights"]
        )
        return X