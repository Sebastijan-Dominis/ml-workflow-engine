"""Categorical missing-value imputation components."""

from ..base import PipelineComponent


class FillCategoricalMissing(PipelineComponent):
    """Fill missing values in categorical columns with the string "missing".

    This transformer coerces values to string and replaces NA values with
    the literal "missing" to keep downstream categorical encoders robust.
    """

    def __init__(self, categorical_features):
        """Store categorical feature names targeted for imputation.

        Args:
            categorical_features: Column names to impute.

        Returns:
            None: Initializes transformer state.
        """

        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        """Return ``self`` because this transformer is stateless.

        Args:
            X: Input features (unused).
            y: Optional target (unused).

        Returns:
            FillCategoricalMissing: Fitted transformer instance.
        """

        return self

    def transform(self, X):
        """Replace missing categorical values with the literal ``"missing"``.

        Args:
            X: Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with imputed categorical columns.
        """

        X = X.copy()
        for col in self.categorical_features:
            X[col] = X[col].astype(str).fillna("missing")
        return X
