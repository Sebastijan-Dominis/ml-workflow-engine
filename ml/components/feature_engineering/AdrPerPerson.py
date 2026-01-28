from .base import FeatureOperator
from ..base import SklearnFeatureMixin

class AdrPerPerson(FeatureOperator, SklearnFeatureMixin):
    output_features = ["adr_per_person"]

    def transform(self, X):
        denom = (
            X["adults"] + X["children"] + X["babies"]
        ).replace(0, 1)
        X = X.copy()
        X["adr_per_person"] = X["adr"] / denom
        return X