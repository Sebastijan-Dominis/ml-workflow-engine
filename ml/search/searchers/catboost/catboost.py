"""CatBoost searcher implementation using multi-step pipeline orchestration."""

import logging
from pathlib import Path

from ml.config.schemas.model_cfg import SearchModelConfig
from ml.search.searchers.base import Searcher
from ml.search.searchers.catboost.pipeline.context import SearchContext
from ml.search.searchers.catboost.pipeline.steps.broad_search import BroadSearchStep
from ml.search.searchers.catboost.pipeline.steps.narrow_search import NarrowSearchStep
from ml.search.searchers.catboost.pipeline.steps.preparation import PreparationStep
from ml.search.searchers.catboost.search_results_creator import create_search_results
from ml.search.searchers.output import SearchOutput
from ml.utils.pipeline_core.runner import PipelineRunner

logger = logging.getLogger(__name__)

class CatBoostSearcher(Searcher):
    """Run preparation, broad search, and optional narrow search phases."""

    def search(
        self,
        model_cfg: SearchModelConfig,
        *,
        strict: bool,
        failure_management_dir: Path,
    ) -> SearchOutput:
        """Execute CatBoost hyperparameter search and return structured output.

        Args:
            model_cfg: Validated search configuration.
            strict: Whether loading and validation stages should fail strictly.
            failure_management_dir: Directory for persisting failure-management artifacts.

        Returns:
            Search output containing search results, lineage, and run metadata.
        """
        ctx = SearchContext(
            model_cfg=model_cfg,
            strict=strict,
            failure_management_dir=failure_management_dir
        )
        runner = PipelineRunner(steps=[
            PreparationStep(),
            BroadSearchStep(),
            NarrowSearchStep()
        ])
        ctx = runner.run(ctx)
        search_results = create_search_results(ctx)
        return SearchOutput(
            search_results=search_results,
            feature_lineage=ctx.require_feature_lineage,
            pipeline_hash=ctx.require_pipeline_hash,
            scoring_method=ctx.require_scoring,
            splits_info=ctx.require_splits_info
        )
