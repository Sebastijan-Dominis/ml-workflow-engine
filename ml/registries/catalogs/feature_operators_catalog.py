"""Registry of feature engineering operator classes available to pipelines."""

from ml.components.feature_engineering.adr_per_person import AdrPerPerson
from ml.components.feature_engineering.arrival_date import ArrivalDate
from ml.components.feature_engineering.arrival_season import ArrivalSeason
from ml.components.feature_engineering.total_stay import TotalStay

# For future use
from ml.components.feature_engineering.pit_operator import PITOperator
# At the moment, customer_id is missing, so we cannot implement PITOperator for tabular features, but we can keep it here for when we have the necessary data

FEATURE_OPERATORS = {
    "TotalStay": TotalStay,
    "AdrPerPerson": AdrPerPerson,
    "ArrivalSeason": ArrivalSeason,
    "ArrivalDate": ArrivalDate,
}
