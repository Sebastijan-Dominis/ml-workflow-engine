"""Feature operator for deriving seasonal arrival categories."""

from ml.components.base import SklearnFeatureMixin
from ml.components.feature_engineering.base import FeatureOperator

class ArrivalSeason(FeatureOperator, SklearnFeatureMixin):
    """Map arrival week numbers to coarse seasonal labels."""

    output_features = ["arrival_season"]

    def transform(self, X):
        """Create ``arrival_season`` using week-number based season buckets.

        Args:
            X: Input dataframe containing `arrival_date_week_number`.

        Returns:
            pd.DataFrame: Dataframe including derived `arrival_season`.
        """

        if not hasattr(self, "n_features_in_"):
            self.fit(X)

        def week_to_season(w):
            """Map ISO week number to seasonal bucket label.

            Args:
                w: ISO week number.

            Returns:
                str: Seasonal label.
            """

            if 10 <= w <= 21:
                return "Spring"
            elif 22 <= w <= 34:
                return "Summer"
            elif 35 <= w <= 47:
                return "Fall"
            else:
                return "Winter"

        X = X.copy()
        X["arrival_season"] = X["arrival_date_week_number"].apply(week_to_season)
        return X
