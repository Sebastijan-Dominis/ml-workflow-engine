from ml.search.searchers.catboost.catboost import SearchCatboost

SEARCH_REGISTRY = {
    "catboost": SearchCatboost,
}