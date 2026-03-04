"""Registry mapping algorithm families to hyperparameter searchers."""

from ml.search.searchers.catboost.catboost import CatBoostSearcher

SEARCHERS = {
    "catboost": CatBoostSearcher,
}
