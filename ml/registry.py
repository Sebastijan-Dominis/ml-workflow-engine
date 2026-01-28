from catboost import CatBoostClassifier
from ml.components.schema_validation.validator import SchemaValidator
from ml.components.imputation.categorical import FillCategoricalMissing
from ml.components.feature_engineering.base import FeatureEngineer
from ml.components.feature_selection.selector import FeatureSelector
from ml.components.feature_engineering.TotalStay import TotalStay
from ml.components.feature_engineering.AdrPerPerson import AdrPerPerson
from ml.components.feature_engineering.ArrivalSeason import ArrivalSeason

FEATURE_OPERATORS = {
    "TotalStay": TotalStay,
    "AdrPerPerson": AdrPerPerson,
    "ArrivalSeason": ArrivalSeason,
}

PIPELINE_COMPONENTS = {
    "SchemaValidator": SchemaValidator,
    "FillCategoricalMissing": FillCategoricalMissing,
    "FeatureEngineer": FeatureEngineer,
    "FeatureSelector": FeatureSelector,
    "Model": None,  # Placeholder for model component
}

MODEL_REGISTRY = {
    "CatBoostClassifier": CatBoostClassifier
}