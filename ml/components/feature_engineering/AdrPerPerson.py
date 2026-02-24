from ml.components.base import SklearnFeatureMixin
from ml.components.feature_engineering.base import FeatureOperator


class AdrPerPerson(FeatureOperator, SklearnFeatureMixin):
    output_features = ["adr_per_person"]

    def transform(self, X):
        if not hasattr(self, "n_features_in_"):
            self.fit(X)

        denom = (
            X["adults"] + X["children"] + X["babies"]
        ).replace(0, 1)
        X = X.copy()
        X["adr_per_person"] = X["adr"] / denom
        return X