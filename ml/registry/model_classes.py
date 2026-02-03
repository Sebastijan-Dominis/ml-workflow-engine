from catboost import CatBoostClassifier, CatBoostRegressor

MODEL_CLASS_REGISTRY = {
    "CatBoostClassifier": CatBoostClassifier,
    "CatBoostRegressor": CatBoostRegressor
}