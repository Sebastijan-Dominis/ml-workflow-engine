import logging
from pathlib import Path

from ml.config.hashing import compute_config_hash
from ml.search.searchers.catboost.pipeline.context import SearchContext
from ml.utils.features.cat_features import get_cat_features
from ml.utils.features.loading.schemas import load_schemas
from ml.utils.features.loading.X_and_y import load_X_and_y
from ml.utils.features.validation import validate_model_feature_pipeline_contract
from ml.utils.loaders import load_yaml
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class PreparationStep(PipelineStep[SearchContext]):
    name = "preparation"

    def before(self, ctx: SearchContext) -> None:
        logger.debug("Starting preparation step.")

    def after(self, ctx: SearchContext) -> None:
        logger.debug("Completed preparation step.")
    
    def run(self, ctx: SearchContext) -> SearchContext:
        X_train, y_train, lineage_train = load_X_and_y(ctx.model_cfg, keys=["X_train", "y_train"], snapshot_selection=None, strict=ctx.strict)
        input_schema, derived_schema = load_schemas(ctx.model_cfg)

        pipeline_path = Path(f"{ctx.model_cfg.pipeline.path}").resolve()
        pipeline_cfg = load_yaml(pipeline_path)
        pipeline_hash = compute_config_hash(pipeline_cfg)

        cat_features = get_cat_features(input_schema, derived_schema)

        validate_model_feature_pipeline_contract(
            ctx.model_cfg,
            pipeline_cfg,
            cat_features
        )
        
        ctx.X_train = X_train
        ctx.y_train = y_train
        ctx.pipeline_cfg = pipeline_cfg
        ctx.input_schema = input_schema
        ctx.derived_schema = derived_schema
        ctx.cat_features = cat_features
        ctx.pipeline_hash = pipeline_hash
        ctx.feature_lineage = lineage_train

        return ctx