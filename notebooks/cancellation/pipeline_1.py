from catboost import CatBoostClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

def create_pipeline(segmented_features = []):
    categorical_features = [
        "hotel",
        "market_segment",
        "meal",
        "distribution_channel",
        "customer_type",
        "deposit_type",
        "arrival_date_month",
        "country",
        "agent",
    ]

    categorical_features = [f for f in categorical_features if f not in segmented_features]

    numerical_features = [
        "lead_time",
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "days_in_waiting_list",
        "adr",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "is_repeated_guest",
    ]

    required_features = categorical_features + numerical_features

    # For CatBoost, include engineered categorical features as well
    cat_features = categorical_features + ["arrival_season"]


    class SchemaValidator(BaseEstimator, TransformerMixin):
        """Validate that incoming DataFrame contains required columns.

        Args:
            required_columns (list): List of column names expected in the input.
        """

        def __init__(self, required_columns):
            self.required_columns = required_columns

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            missing = [c for c in self.required_columns if c not in X.columns]
            if missing:
                raise ValueError(f"Model input schema violation. Missing columns: {missing}")
            return X


    class FillCategoricalMissing(BaseEstimator, TransformerMixin):
        """Fill missing values in categorical columns with the string "missing".

        This transformer coerces values to string and replaces NA values with
        the literal "missing" to keep downstream categorical encoders robust.
        """

        def __init__(self, categorical_columns):
            self.categorical_columns = categorical_columns

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X = X.copy()
            for col in self.categorical_columns:
                X[col] = X[col].astype(str).fillna("missing")
            return X


    class FeatureEngineer(BaseEstimator, TransformerMixin):
        """Create derived features used by the cancellation model.

        Attributes:
            created_columns (list): Column names produced by this transformer.
        """

        created_columns = ["total_stay", "adr_per_person", "arrival_season"]

        def fit(self, X, y=None):
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X):
            X = X.copy()
            X["total_stay"] = X["stays_in_weekend_nights"] + X["stays_in_week_nights"]
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

            X["arrival_season"] = X["arrival_date_week_number"].apply(week_to_season)
            return X


    class FeatureSelector(BaseEstimator, TransformerMixin):
        """Select a fixed set of columns for model training.

        The transformer validates presence of selected columns and returns a
        DataFrame with only the requested columns.
        """

        def __init__(self, selected_columns):
            self.selected_columns = selected_columns

        def fit(self, X, y=None):
            self.n_features_in_ = X.shape[1]  # scikit-learn expects this
            return self

        def transform(self, X):
            # select only required columns, ignore extras
            missing = [c for c in self.selected_columns if c not in X.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            return X[self.selected_columns]
        
    model = CatBoostClassifier(
        # Basic hyperparameters
        iterations=1,       # number of boosting rounds - lower for segments with less data
        task_type='GPU',      # enable GPU
        devices='0',          # GPU device
        verbose=100,          # print progress every 100 iterations
        random_state=42,
        cat_features=cat_features
    )

    steps = [
        ("schema_validator", SchemaValidator(required_features)),
        ("fill_categorical_missing", FillCategoricalMissing(categorical_features)),
        ("feature_engineer", FeatureEngineer()),
        ("feature_selector", FeatureSelector(required_features + FeatureEngineer.created_columns)),
        ("model", model),
    ]

    pipeline = Pipeline(steps)
    
    return pipeline