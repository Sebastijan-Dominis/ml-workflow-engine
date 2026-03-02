"""Point-in-time-safe feature operator for grouped cumulative aggregations."""

import pandas as pd

from ml.components.feature_engineering.base import FeatureOperator, SklearnFeatureMixin


# Currently not used. Optimize better if needed in the future.
class PITOperator(FeatureOperator, SklearnFeatureMixin):
    """Generic PIT-safe operator for cumulative/aggregate features."""

    def __init__(self, groupby_cols, agg_col, agg_func, feature_name):
        """Configure grouped aggregation inputs and generated feature metadata.

        Args:
            groupby_cols: Columns used for grouping.
            agg_col: Aggregation source column.
            agg_func: Aggregation function.
            feature_name: Name of derived feature column.

        Returns:
            None: Initializes operator configuration.
        """

        self.groupby_cols = groupby_cols
        self.agg_col = agg_col
        self.agg_func = agg_func
        self.feature_name = feature_name
        self.output_features = [feature_name]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate lagged grouped cumulative aggregate without leakage.

        Args:
            X: Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with derived PIT-safe feature column.
        """
        if not hasattr(self, "n_features_in_"):
            self.fit(X)
        
        X = X.sort_values(self.groupby_cols + ['arrival_datetime'])
        X[self.feature_name] = (
            X.groupby(self.groupby_cols)[self.agg_col]
             .expanding()
             .agg(self.agg_func)
             .shift(1)
             .reset_index(level=0, drop=True)
        )

        # Fill NaNs for first rows (no history)
        if X[self.feature_name].isnull().any():
            X[self.feature_name] = X[self.feature_name].fillna(X[self.agg_col].mean())

        return X
