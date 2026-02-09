from typing import Any

from ml.search.searchers.catboost.pipeline.context import SearchContext
from ml.search.utils.model_params_extraction import extract_model_params


def create_search_results(ctx: SearchContext) -> dict[str, Any]:
    best_pipeline_params = ctx.best_params_1 if ctx.require_narrow_disabled else ctx.require_best_params
    best_model_params = extract_model_params(best_pipeline_params)
    search_results =  {
        "best_pipeline_params": best_pipeline_params,
        "best_model_params": best_model_params,
        "phases": {
            "broad": ctx.broad_result,
        }
    }
    if not ctx.require_narrow_disabled:
        search_results["phases"]["narrow"] = ctx.narrow_result
    return search_results