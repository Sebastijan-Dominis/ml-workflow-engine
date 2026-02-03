from ml.components.feature_engineering.TotalStay import TotalStay
from ml.components.feature_engineering.AdrPerPerson import AdrPerPerson
from ml.components.feature_engineering.ArrivalSeason import ArrivalSeason
# For future use
from ml.components.feature_engineering.PITOperator import PITOperator

FEATURE_OPERATORS = {
    "TotalStay": TotalStay,
    "AdrPerPerson": AdrPerPerson,
    "ArrivalSeason": ArrivalSeason,
}
