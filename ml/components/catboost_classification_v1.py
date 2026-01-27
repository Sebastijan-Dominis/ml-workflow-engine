"""Model-specific pipeline components for the ``cancellation_v1`` model.

This module exposes lists of feature names and several scikit-learn
compatible transformer classes used to build preprocessing pipelines for
the cancellation prediction model:

- ``categorical_features``: categorical column names
- ``numerical_features``: numerical column names
- ``required_features``: the superset of features required by the model
- ``cat_features``: list of categorical features passed to CatBoost

Transformers:
- ``SchemaValidator``: validates presence of required columns
- ``FillCategoricalMissing``: fills missing categorical values with "missing"
- ``FeatureEngineer``: creates derived features used by the model
- ``FeatureSelector``: selects the final set of columns for training
"""

from sklearn.base import BaseEstimator, TransformerMixin

class SchemaValidator(BaseEstimator, TransformerMixin):
    """Validate that incoming DataFrame contains required columns.

    Args:
        required_features (list): List of column names expected in the input.
    """

    def __init__(self, required_features):
        self.required_features = required_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing = [c for c in self.required_features if c not in X.columns]
        if missing:
            raise ValueError(f"Model input schema violation. Missing columns: {missing}")
        return X


class FillCategoricalMissing(BaseEstimator, TransformerMixin):
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


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create derived features used by the cancellation model.

    Attributes:
        engineered_features (list): Column names produced by this transformer.
    """
    def __init__(self, engineered_features=[]):
        self.engineered_features = engineered_features

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()

        if "total_stay" in self.engineered_features:
            X["total_stay"] = X["stays_in_weekend_nights"] + X["stays_in_week_nights"]
    
        if "adr_per_person" in self.engineered_features:
            X["adr_per_person"] = X["adr"] / (X["adults"] + X["children"] + X["babies"]).replace(0, 1)

        def week_to_season(week):
            if 10 <= week <= 21:
                return "Spring"
            elif 22 <= week <= 34:
                return "Summer"
            elif 35 <= week <= 47:
                return "Fall"
            else:
                return "Winter"

        if "arrival_season" in self.engineered_features:
            X["arrival_season"] = X["arrival_date_week_number"].apply(week_to_season)
        
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select a fixed set of columns for model training.

    The transformer validates presence of selected columns and returns a
    DataFrame with only the requested columns.
    """

    def __init__(self, selected_features):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]  # scikit-learn expects this
        return self

    def transform(self, X):
        # select only required columns, ignore extras
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return X[self.selected_features]