import logging

import numpy as np

from ml.exceptions import ConfigError, SearchError, UserError
from ml.search.searchers.catboost.model import prepare_model
from ml.search.searchers.catboost.pipeline.context import SearchContext
from ml.search.utils.failure_management.save_broad import save_broad
from ml.search.utils.randomized_search import perform_randomized_search
from ml.utils.catboost.build_pipeline_with_model import \
    build_pipeline_with_model
from ml.utils.loaders import load_json
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class BroadSearchStep(PipelineStep[SearchContext]):
    name = "broad_search"

    def before(self, ctx: SearchContext) -> None:
        logger.info("Starting broad search step.")
        
    def after(self, ctx: SearchContext) -> None:
        logger.info("Completed broad search step.")

    def run(self, ctx: SearchContext) -> SearchContext:
        broad_info_path = ctx.failure_management_dir / "broad_info.json"
        best_broad_info = load_json(broad_info_path, strict=False)
        if best_broad_info:
            logger.info(f"Found existing best broad info for experiment {ctx.failure_management_dir.name} at {broad_info_path}. Skipping broad search and using these params for narrow search.")
            broad_result = best_broad_info.get("broad_result")
            best_params_1 = best_broad_info.get("best_params_1")
            if broad_result is None or best_params_1 is None:
                msg = f"Broad info file {broad_info_path} is missing required keys. Expected keys: 'broad_result' and 'best_params_1'."
                logger.error(msg)
                raise UserError(msg)
            ctx.broad_result = broad_result
            ctx.best_params_1 = best_params_1
            return ctx

        model_1 = prepare_model(
            ctx.model_cfg, 
            search_phase="broad",
            cat_features=ctx.require_cat_features, 
            class_weights=ctx.class_weights
        )

        pipeline_1 = build_pipeline_with_model(
            model_cfg=ctx.model_cfg,
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

        logger.info("Broad search param combinations: %d", np.prod([len(v) for v in broad_param_distributions.values()]))

        try:
            broad_result = perform_randomized_search(
                pipeline_1,
                X_train=ctx.require_X_train,
                y_train=ctx.require_y_train,
                param_distributions=broad_param_distributions,
                model_cfg=ctx.model_cfg,
                scoring=ctx.require_scoring,
                search_phase="broad"
            )
        except Exception as e:
            msg = f"Broad hyperparameter search failed for problem={ctx.model_cfg.problem} segment={ctx.model_cfg.segment.name} version={ctx.model_cfg.version}: {e}"
            logger.error(msg)
            raise SearchError(msg) from e

        best_params_1 = broad_result["best_params"]

        ctx.broad_result = broad_result
        ctx.best_params_1 = best_params_1

        save_broad(
            broad_result=broad_result,
            best_params_1=best_params_1,
            tgt_file=broad_info_path
        )

        return ctx