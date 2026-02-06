import logging

from ml.search.params.catboost.refinement import prepare_narrow_params
from ml.search.params.catboost.validation import validate_param_value
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

class NarrowSearchStep(PipelineStep[SearchContext]):
    name = "narrow_search"

    def before(self, ctx: SearchContext) -> None:
        logger.debug("Starting narrow search step.")

    def after(self, ctx: SearchContext) -> None:
        logger.debug("Completed narrow search step.")

    def run(self, ctx: SearchContext) -> SearchContext:
        if not ctx.model_cfg.search.narrow.enabled:
            logger.info("Narrow search is disabled in the model config for problem=%s segment=%s version=%s. Skipping narrow search step.",
                ctx.model_cfg.problem, ctx.model_cfg.segment.name, ctx.model_cfg.version)
            ctx.narrow_disabled = True
            return ctx
        
        ctx.narrow_disabled = False

        narrow_param_cfg = ctx.model_cfg.search.narrow.param_configurations
        if not narrow_param_cfg or not narrow_param_cfg.model_dump(exclude_none=True):
            msg = f"No narrow search param_configurations defined in the model config for problem={ctx.model_cfg.problem} segment={ctx.model_cfg.segment.name} version={ctx.model_cfg.version}."
            logger.error(msg)
            raise ConfigError(msg)

        narrow_param_distributions = prepare_narrow_params(ctx.require_best_params_1, narrow_param_cfg, ctx.model_cfg.search.hardware.task_type.value)
        for param, values in narrow_param_distributions.items():
            base_param_name = param.replace("Model__", "")
            for v in values:
                validate_param_value(base_param_name, v, str(ctx.model_cfg.search.hardware.task_type.value).upper())

        model_2 = prepare_model(ctx.model_cfg, "narrow", ctx.require_cat_features)

        pipeline_2 = build_pipeline_with_model(
            pipeline_cfg=ctx.require_pipeline_cfg,
            input_schema=ctx.require_input_schema,
            derived_schema=ctx.require_derived_schema,
            model=model_2
        )

        logger.info("Starting narrow hyperparameter search | problem=%s segment=%s version=%s",
            ctx.model_cfg.problem, ctx.model_cfg.segment.name, ctx.model_cfg.version)

        logger.debug("Narrow search param combinations: %d", np.prod([len(v) for v in narrow_param_distributions.values()]))

        try:
            narrow_result = perform_randomized_search(
                pipeline_2,
                ctx.require_X_train,
                ctx.require_y_train,
                narrow_param_distributions,
                ctx.model_cfg,
                search_type="narrow"
            )
        except Exception as e:
            msg = f"Narrow hyperparameter search failed for problem={ctx.model_cfg.problem} segment={ctx.model_cfg.segment.name} version={ctx.model_cfg.version}: {e}"
            logger.error(msg)
            raise SearchError(msg) from e

        best_params = narrow_result["best_params"]

        ctx.narrow_result = narrow_result
        ctx.best_params = best_params

        return ctx