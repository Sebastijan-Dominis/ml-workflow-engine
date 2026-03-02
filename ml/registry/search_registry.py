"""Registry mapping algorithm families to hyperparameter searchers."""

from ml.search.searchers.catboost.catboost import SearchCatboost

SEARCHERS = {
    "catboost": SearchCatboost,
}