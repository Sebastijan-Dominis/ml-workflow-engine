import logging
logger = logging.getLogger(__name__)
from pathlib import Path

from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import FreezeContext
from ml.utils.pipeline_core.step import PipelineStep
from ml.feature_freezing.freeze_strategies.tabular.io import load_and_hash_data
from ml.feature_freezing.freeze_strategies.tabular.segmentation import apply_segmentation
from ml.feature_freezing.freeze_strategies.tabular.validation import validate_min_rows, validate_min_class_count, validate_include_exclude_columns
from ml.feature_freezing.utils.operators import validate_operators

class IngestionStep(PipelineStep[FreezeContext]):
    name = "ingestion"

    def before(self, ctx: FreezeContext) -> None:
        logger.debug("Starting data ingestion step.")

    def after(self, ctx: FreezeContext) -> None:
        logger.debug("Completed data ingestion step.")
    
    def run(self, ctx: FreezeContext) -> FreezeContext:
        data, data_hash = load_and_hash_data(
            Path(ctx.config.data.path),
            ctx.config.data.format
        )

        data = apply_segmentation(data, ctx.config)

        validate_min_rows(data, ctx.config.min_rows)

        if ctx.config.target.problem_type == "classification":
            validate_min_class_count(
                data[ctx.config.target.name],
                ctx.config.min_class_count
            )

        if ctx.config.operators:
            validate_operators(
                ctx.config.operators.list,
                ctx.config.operators.hash
            )

        validate_include_exclude_columns(ctx.config)

        ctx.data = data
        ctx.data_hash = data_hash

        return ctx