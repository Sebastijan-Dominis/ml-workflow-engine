import logging
logger = logging.getLogger(__name__)
import numpy as np

from ml.search.searchers.catboost.pipeline.context import SearchContext
from ml.utils.pipeline_core.step import PipelineStep
from ml.exceptions import ConfigError, SearchError
from ml.search.searchers.catboost.model import (
    prepare_model,
    build_pipeline_with_model,
)
from ml.search.utils.utils import perform_randomized_search

class BroadSearchStep(PipelineStep[SearchContext]):
    name = "broad_search"

    def before(self, ctx: SearchContext) -> None:
        logger.debug("Starting broad search step.")
        
    def after(self, ctx: SearchContext) -> None:
        logger.debug("Completed broad search step.")

    def run(self, ctx: SearchContext) -> SearchContext:
        model_1 = prepare_model(ctx.model_cfg, "broad", ctx.require_cat_features)

        pipeline_1 = build_pipeline_with_model(
            pipeline_cfg=ctx.require_pipeline_cfg,
            input_schema=ctx.require_input_schema,
            derived_schema=ctx.require_derived_schema,
            model=model_1
        )

        broad_param_distributions_obj = ctx.model_cfg.search.broad.param_distributions

        if not broad_param_distributions_obj or not broad_param_distributions_obj.model_dump(exclude_none=True):
            msg = f"No broad search param_distributions defined in the model config for problem={ctx.model_cfg.problem} segment={ctx.model_cfg.segment.name} version={ctx.model_cfg.version}."
            logger.error(msg)
            raise ConfigError(msg)

        broad_param_distributions = broad_param_distributions_obj.to_flat_dict()

        logger.info("Starting broad hyperparameter search | problem=%s segment=%s version=%s",
            ctx.model_cfg.problem, ctx.model_cfg.segment.name, ctx.model_cfg.version)

        logger.debug("Broad search param combinations: %d", np.prod([len(v) for v in broad_param_distributions.values()]))

        try:
            broad_result = perform_randomized_search(
                pipeline_1,
                ctx.require_X_train,
                ctx.require_y_train,
                broad_param_distributions,
                ctx.model_cfg,
                search_type="broad"
            )
        except Exception as e:
            msg = f"Broad hyperparameter search failed for problem={ctx.model_cfg.problem} segment={ctx.model_cfg.segment.name} version={ctx.model_cfg.version}: {e}"
            logger.error(msg)
            raise SearchError(msg) from e

        best_params_1 = broad_result["best_params"]

        ctx.broad_result = broad_result
        ctx.best_params_1 = best_params_1

        return ctx