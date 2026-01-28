from ..base import PipelineComponent

class FillCategoricalMissing(PipelineComponent):
    """Fill missing values in categorical columns with the string "missing".

    This transformer coerces values to string and replaces NA values with
    the literal "missing" to keep downstream categorical encoders robust.
    """

    def __init__(self, categorical_features):
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.categorical_features:
            X[col] = X[col].astype(str).fillna("missing")
        return X