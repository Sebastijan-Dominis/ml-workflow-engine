"""Registry mapping model class names to concrete estimator classes."""

from catboost import CatBoostClassifier, CatBoostRegressor

MODEL_CLASS_REGISTRY = {
    "CatBoostClassifier": CatBoostClassifier,
    "CatBoostRegressor": CatBoostRegressor
}
