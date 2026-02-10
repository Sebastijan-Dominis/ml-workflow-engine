import logging
from typing import Any

from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.search.searchers.base import Searcher
from ml.search.searchers.catboost.pipeline.context import SearchContext
from ml.search.searchers.catboost.pipeline.steps.broad_search import BroadSearchStep
from ml.search.searchers.catboost.pipeline.steps.narrow_search import NarrowSearchStep
from ml.search.searchers.catboost.pipeline.steps.preparation import PreparationStep
from ml.search.searchers.catboost.search_results_creator import create_search_results
from ml.utils.pipeline_core.runner import PipelineRunner

logger = logging.getLogger(__name__)

class SearchCatboost(Searcher):
    def search(self, model_cfg: SearchModelConfig, strict: bool) -> tuple[dict[str, Any], list[dict], str]:
        ctx = SearchContext(model_cfg=model_cfg, strict=strict)
        runner = PipelineRunner(steps=[
            PreparationStep(),
            BroadSearchStep(),
            NarrowSearchStep()
        ])
        ctx = runner.run(ctx)
        search_results = create_search_results(ctx)
        return search_results, ctx.require_feature_lineage, ctx.require_pipeline_hash