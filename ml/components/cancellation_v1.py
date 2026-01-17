from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------
# Step 1 - Define features
# -------------------------------------------
categorical_features = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "customer_type", "deposit_type", "arrival_date_month", "country", "agent"
]

numerical_features = [
    "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "previous_cancellations",
    "previous_bookings_not_canceled", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests",
    "is_repeated_guest"
]

required_features = categorical_features + numerical_features

# For CatBoost, we can include "arrival_season" as a categorical feature, since CatBoost can handle categorical features natively, and arrival_season is an engineered categorical feature.
cat_features = categorical_features + ["arrival_season"]

# ---------------------------------------------------
# Step 2 — Define SchemaValidator
# ---------------------------------------------------
class SchemaValidator(BaseEstimator, TransformerMixin):
    def __init__(self, required_columns):
        self.required_columns = required_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing = [c for c in self.required_columns if c not in X.columns]
        if missing:
            raise ValueError(
                f"Model input schema violation. Missing columns: {missing}"
            )
        return X

# ---------------------------------------------------
# Step 3 - Define the class to fill missing categorical values
# ---------------------------------------------------
class FillCategoricalMissing(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.categorical_columns:
            X[col] = X[col].astype(str).fillna("missing")
        return X

# ---------------------------------------------------
# Step 4 — Feature engineering
# ---------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    created_columns = ['total_stay', 'adr_per_person', 'arrival_season']

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        X['total_stay'] = X['stays_in_weekend_nights'] + X['stays_in_week_nights']
        X['adr_per_person'] = X['adr'] / (X['adults'] + X['children'] + X['babies']).replace(0,1)
        # simple season mapping
        def week_to_season(week):
            if 10 <= week <= 21:
                return 'Spring'
            elif 22 <= week <= 34:
                return 'Summer'
            elif 35 <= week <= 47:
                return 'Fall'
            else:
                return 'Winter'
        X['arrival_season'] = X['arrival_date_week_number'].apply(week_to_season)
        return X

# ---------------------------------------------------
# Step 5 - Define feature selector
# ---------------------------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
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