"""Registry of feature engineering operator classes available to pipelines."""

from ml.components.feature_engineering.AdrPerPerson import AdrPerPerson
from ml.components.feature_engineering.ArrivalDate import ArrivalDate
from ml.components.feature_engineering.ArrivalSeason import ArrivalSeason
from ml.components.feature_engineering.TotalStay import TotalStay

# For future use
# At the moment, customer_id is missing, so we cannot implement PITOperator for tabular features, but we can keep it here for when we have the necessary data

FEATURE_OPERATORS = {
    "TotalStay": TotalStay,
    "AdrPerPerson": AdrPerPerson,
    "ArrivalSeason": ArrivalSeason,
    "ArrivalDate": ArrivalDate,
}
