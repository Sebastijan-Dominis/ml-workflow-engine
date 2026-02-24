import pandas as pd

from ml.components.base import SklearnFeatureMixin
from ml.components.feature_engineering.base import FeatureOperator

month_map = {
    'January': 1,'February': 2,'March': 3,'April': 4,'May': 5,'June': 6,
    'July': 7,'August': 8,'September': 9,'October': 10,'November': 11,'December': 12
}

class ArrivalDate(FeatureOperator, SklearnFeatureMixin):
    output_features = ["arrival_date"]

    def transform(self, X):
        if not hasattr(self, "n_features_in_"):
            self.fit(X)

        month_series = X["arrival_date_month"].map(month_map)
        X = X.copy()
        X["arrival_date"] = pd.to_datetime(
            {
                "year": X["arrival_date_year"],
                "month": month_series,
                "day": X["arrival_date_day_of_month"],
            }
        )
        return X