from sklearn.base import BaseEstimator, TransformerMixin

# Define classes and thresholds for classification
# Custom transformer for engineered features
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    # A function to map week number to season
    def week_to_season(self, week):
        if 10 <= week <= 21:
            return 'Spring'
        elif 22 <= week <= 34:
            return 'Summer'
        elif 35 <= week <= 47:
            return 'Fall'
        else:
            return 'Winter'

    def transform(self, X):
        X = X.copy()
        # total_stay
        X['total_stay'] = X['stays_in_weekend_nights'] + X['stays_in_week_nights']
        # adr_per_person
        X['adr_per_person'] = X['adr'] / (X['adults'] + X['children'] + X['babies']).replace(0, 1)
        # arrival_season
        X['arrival_season'] = X['arrival_date_week_number'].apply(self.week_to_season)
        return X
    
    created_columns = ['total_stay', 'adr_per_person', 'arrival_season']
    
# Custom transformer for engineered features
class FeatureEngineerRepeatedGuest(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # total_stay
        X['total_stay'] = X['stays_in_weekend_nights'] + X['stays_in_week_nights']
        # adr_per_person
        X['adr_per_person'] = X['adr'] / (X['adults'] + X['children'] + X['babies']).replace(0, 1)
        return X
    
    created_columns = ['total_stay', 'adr_per_person']

# Schema validator to ensure required columns are present
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

# Define classification thresholds
CANCELLATION_THRESHOLD = 0.4  # Threshold for classifying a booking as cancelled
NO_SHOW_THRESHOLD = 0.6       # Threshold for classifying a booking as no-show
REPEATED_GUEST_THRESHOLD = 0.76  # Threshold for classifying a booking as a repeated guest
ROOM_UPGRADES_THRESHOLD = 0.46  # Threshold for classifying a booking as having room upgrades