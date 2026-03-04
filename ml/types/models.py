"""Type aliases for supported trained model object types."""

from catboost import CatBoostClassifier, CatBoostRegressor

# Update this type alias if additional model types are added in the future
AllowedModels = CatBoostClassifier | CatBoostRegressor
