from typing import Callable

from catboost import CatBoostClassifier, CatBoostRegressor
from prophet import Prophet
from sklearn.pipeline import Pipeline

from ml.runners.training.utils.hashing.helpers import _hash_catboost, _hash_prophet, _hash_sklearn_pipeline

HASH_ARTIFACT_REGISTRY: dict[type, Callable] = {
    CatBoostClassifier: _hash_catboost,
    CatBoostRegressor: _hash_catboost,
    Prophet: _hash_prophet,
    Pipeline: _hash_sklearn_pipeline
}