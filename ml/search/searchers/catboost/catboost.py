import logging
logger = logging.getLogger(__name__)
from typing import Any

from ml.search.searchers.base import BaseSearcher
from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.search.searchers.catboost.pipeline.context import SearchContext
from ml.utils.pipeline_core.runner import PipelineRunner
from ml.search.searchers.catboost.pipeline.steps.preparation import PreparationStep
from ml.search.searchers.catboost.pipeline.steps.broad_search import BroadSearchStep
from ml.search.searchers.catboost.pipeline.steps.narrow_search import NarrowSearchStep

class SearchCatboost(BaseSearcher):
    def search(self, model_cfg: SearchModelConfig) -> dict[str, Any]:
        ctx = SearchContext(model_cfg=model_cfg)
        runner = PipelineRunner(steps=[
            PreparationStep(),
            BroadSearchStep(),
            NarrowSearchStep()
        ])
        ctx = runner.run(ctx)
        search_results =  {
            "best_params": ctx.best_params_1 if ctx.require_narrow_disabled else ctx.require_best_params,
            "phases": {
                "broad": ctx.broad_result,
            }
        }
        if not ctx.require_narrow_disabled:
            search_results["phases"]["narrow"] = ctx.narrow_result
        return search_results