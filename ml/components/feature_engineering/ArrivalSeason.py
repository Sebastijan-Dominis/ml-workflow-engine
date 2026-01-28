from .base import FeatureOperator
from ..base import SklearnFeatureMixin

class ArrivalSeason(FeatureOperator, SklearnFeatureMixin):
    output_features = ["arrival_season"]

    def transform(self, X):
        def week_to_season(w):
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
