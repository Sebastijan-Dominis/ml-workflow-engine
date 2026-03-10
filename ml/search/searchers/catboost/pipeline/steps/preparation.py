"""Preparation pipeline step for CatBoost search orchestration."""

import logging
from pathlib import Path

from ml.config.hashing import compute_model_config_hash
from ml.features.extraction.cat_features import get_cat_features
from ml.features.loading.features_and_target import load_features_and_target
from ml.features.loading.schemas import load_schemas
from ml.features.splitting.splitting import get_splits
from ml.features.transforms.transform_target import transform_target
from ml.features.validation.validate_contract import validate_model_feature_pipeline_contract
from ml.modeling.class_weighting.models import DataStats
from ml.modeling.class_weighting.resolve_class_weighting import resolve_class_weighting
from ml.modeling.class_weighting.resolve_metric import resolve_metric
from ml.modeling.class_weighting.stats_resolver import compute_data_stats
from ml.search.searchers.catboost.pipeline.context import SearchContext
from ml.utils.loaders import load_yaml
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class PreparationStep(PipelineStep[SearchContext]):
    """Load data/schemas, resolve scoring, and prepare search context state."""

    name = "preparation"

    stats: DataStats

    def before(self, ctx: SearchContext) -> None:
        """Emit pre-step log message.

        Args:
            ctx: Search pipeline context.

        Returns:
            None: Emits logging side effect only.
        """
        logger.info("Starting preparation step.")

    def after(self, ctx: SearchContext) -> None:
        """Emit post-step log message.

        Args:
            ctx: Search pipeline context.

        Returns:
            None: Emits logging side effect only.
        """
        logger.info("Completed preparation step.")

    def run(self, ctx: SearchContext) -> SearchContext:
        """Prepare training inputs/config artifacts required by search phases.

        Args:
            ctx: Search pipeline context.

        Returns:
            SearchContext: Updated context with prepared data/config artifacts.

        Raises:
            DataError: Propagated from feature/target loading and validation.
            PipelineContractError: Propagated when model/pipeline contract checks
                fail.

        Notes:
            This step centralizes all shared preparation work so broad and narrow
            phases consume the same validated inputs and derived artifacts.

        Side Effects:
            Reads data/config files from disk and mutates multiple context fields
            required by downstream search steps.
        """
        X, y, lineage = load_features_and_target(
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
        input_schema, derived_schema = load_schemas(ctx.model_cfg, lineage)

        pipeline_path = Path(f"{ctx.model_cfg.pipeline.path}").resolve()
        pipeline_cfg = load_yaml(pipeline_path)
        pipeline_hash = compute_model_config_hash(pipeline_cfg)

        cat_features = get_cat_features(ctx.model_cfg, input_schema, derived_schema)

        validate_model_feature_pipeline_contract(
            ctx.model_cfg,
            pipeline_cfg,
            cat_features
        )

        stats = None
        if ctx.model_cfg.task.type == "classification":
            stats = compute_data_stats(splits.y_train)

        scoring = resolve_metric(ctx.model_cfg, stats)
        ctx.scoring = scoring

        if stats is not None:
            class_weights = resolve_class_weighting(ctx.model_cfg, stats, library="catboost")
            ctx.class_weights = class_weights

        y_train = transform_target(
            splits.y_train,
            transform_config=ctx.model_cfg.target.transform,
            split_name="train"
        )

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
