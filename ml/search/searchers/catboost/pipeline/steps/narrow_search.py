import logging

import numpy as np

from ml.exceptions import ConfigError, SearchError, UserError
from ml.search.params.catboost.refinement import prepare_narrow_params
from ml.search.params.catboost.validation import validate_param_value
from ml.search.searchers.catboost.model import prepare_model
from ml.search.searchers.catboost.pipeline.context import SearchContext
from ml.search.utils.failure_management.save_narrow import save_narrow
from ml.search.utils.randomized_search import perform_randomized_search
from ml.utils.catboost.build_pipeline_with_model import \
    build_pipeline_with_model
from ml.utils.loaders import load_json
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class NarrowSearchStep(PipelineStep[SearchContext]):
    name = "narrow_search"

    def before(self, ctx: SearchContext) -> None:
        logger.info("Starting narrow search step.")

    def after(self, ctx: SearchContext) -> None:
        logger.info("Completed narrow search step.")

    def run(self, ctx: SearchContext) -> SearchContext:
        narrow_info_path = ctx.failure_management_dir / "narrow_info.json"
        narrow_info = load_json(narrow_info_path, strict=False)
        if narrow_info:
            logger.info(f"Found existing narrow info for experiment {ctx.failure_management_dir.name} at {narrow_info_path}. Skipping narrow search.")
            narrow_result = narrow_info.get("narrow_result")
            best_params = narrow_info.get("best_params")
            if narrow_result is None or best_params is None:
                msg = f"Narrow info file {narrow_info_path} is missing required keys. Expected keys: 'narrow_result' and 'best_params'."
                logger.error(msg)
                raise UserError(msg)
            ctx.narrow_result = narrow_result
            ctx.best_params = best_params
            ctx.narrow_disabled = False
            return ctx

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

        narrow_param_distributions = prepare_narrow_params(
            best_params=ctx.require_best_params_1, 
            narrow_params_cfg=narrow_param_cfg, 
            task_type=ctx.model_cfg.search.hardware.task_type.value
        )

        for param, values in narrow_param_distributions.items():
            base_param_name = param.replace("Model__", "")
            for v in values:
                validate_param_value(base_param_name, v, str(ctx.model_cfg.search.hardware.task_type.value).upper())

        model_2 = prepare_model(
            ctx.model_cfg, 
            search_phase="narrow",
            cat_features=ctx.require_cat_features, 
            class_weights=ctx.class_weights
        )

        pipeline_2 = build_pipeline_with_model(
            model_cfg=ctx.model_cfg,
            pipeline_cfg=ctx.require_pipeline_cfg,
            input_schema=ctx.require_input_schema,
            derived_schema=ctx.require_derived_schema,
            model=model_2
        )

        logger.info("Narrow search param combinations: %d", np.prod([len(v) for v in narrow_param_distributions.values()]))

        try:
            narrow_result = perform_randomized_search(
                pipeline_2,
                X_train=ctx.require_X_train,
                y_train=ctx.require_y_train,
                param_distributions=narrow_param_distributions,
                model_cfg=ctx.model_cfg,
                scoring=ctx.require_scoring,
                search_phase="narrow"
            )
        except Exception as e:
            msg = f"Narrow hyperparameter search failed for problem={ctx.model_cfg.problem} segment={ctx.model_cfg.segment.name} version={ctx.model_cfg.version}: {e}"
            logger.error(msg)
            raise SearchError(msg) from e

        best_params = narrow_result["best_params"]

        ctx.narrow_result = narrow_result
        ctx.best_params = best_params

        save_narrow(
            narrow_result=narrow_result,
            best_params=best_params,
            tgt_file=narrow_info_path
        )

        return ctx