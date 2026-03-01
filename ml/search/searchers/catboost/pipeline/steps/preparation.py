import logging
from pathlib import Path

from ml.config.hashing import compute_config_hash
from ml.search.searchers.catboost.pipeline.context import SearchContext
from ml.utils.experiments.class_weights.models import DataStats
from ml.utils.experiments.class_weights.resolve_class_weighting import \
    resolve_class_weighting
from ml.utils.experiments.class_weights.resolve_metric import resolve_metric
from ml.utils.experiments.class_weights.stats_resolver import \
    compute_data_stats
from ml.utils.features.cat_features import get_cat_features
from ml.utils.features.loading.schemas import load_schemas
from ml.utils.features.loading.X_and_y import load_X_and_y
from ml.utils.features.splitting.splitting import get_splits
from ml.utils.features.transform_target import transform_target
from ml.utils.features.validation.validate_contract import \
    validate_model_feature_pipeline_contract
from ml.utils.loaders import load_yaml
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class PreparationStep(PipelineStep[SearchContext]):
    name = "preparation"

    stats: DataStats

    def before(self, ctx: SearchContext) -> None:
        logger.debug("Starting preparation step.")

    def after(self, ctx: SearchContext) -> None:
        logger.debug("Completed preparation step.")
    
    def run(self, ctx: SearchContext) -> SearchContext:
        X, y, lineage = load_X_and_y(
            ctx.model_cfg, 
            snapshot_selection=None, 
            strict=ctx.strict
        )
        splits, splits_info = get_splits(
            X,
            y,
            split_cfg=ctx.model_cfg.split,
            data_type=ctx.model_cfg.data_type,
            task_cfg=ctx.model_cfg.task
        )
        input_schema, derived_schema = load_schemas(ctx.model_cfg)

        pipeline_path = Path(f"{ctx.model_cfg.pipeline.path}").resolve()
        pipeline_cfg = load_yaml(pipeline_path)
        pipeline_hash = compute_config_hash(pipeline_cfg)

        cat_features = get_cat_features(ctx.model_cfg, input_schema, derived_schema)

        validate_model_feature_pipeline_contract(
            ctx.model_cfg,
            pipeline_cfg,
            cat_features
        )

        stats = compute_data_stats(splits.y_train)
        logger.info("Data stats | n_samples=%d class_counts=%s minority_ratio=%.4f",
            stats.n_samples, stats.class_counts, stats.minority_ratio)
        
        scoring = resolve_metric(ctx.model_cfg, stats)
        ctx.scoring = scoring
        
        class_weights = resolve_class_weighting(ctx.model_cfg, stats, library="catboost")
        ctx.class_weights = class_weights

        y_train = transform_target(splits.y_train, ctx.model_cfg.target.transform)

        ctx.X_train = splits.X_train
        ctx.y_train = y_train
        ctx.splits_info = splits_info
        ctx.pipeline_cfg = pipeline_cfg
        ctx.input_schema = input_schema
        ctx.derived_schema = derived_schema
        ctx.cat_features = cat_features
        ctx.pipeline_hash = pipeline_hash
        ctx.feature_lineage = lineage

        return ctx