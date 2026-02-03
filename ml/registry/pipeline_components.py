from ml.components.schema_validation.validator import SchemaValidator
from ml.components.imputation.categorical import FillCategoricalMissing
from ml.components.feature_engineering.base import FeatureEngineer
from ml.components.feature_selection.selector import FeatureSelector

PIPELINE_COMPONENTS = {
    "SchemaValidator": SchemaValidator,
    "FillCategoricalMissing": FillCategoricalMissing,
    "FeatureEngineer": FeatureEngineer,
    "FeatureSelector": FeatureSelector,
    "Model": None,  # Placeholder for model component
}