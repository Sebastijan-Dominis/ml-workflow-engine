import logging
from pathlib import Path

from ml.feature_freezing.freeze_strategies.tabular.io import load_data_with_loader_validation_hash
from ml.utils.loaders import load_json
from ml.utils.data.validate_dataset import validate_dataset
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import FreezeContext
from ml.feature_freezing.freeze_strategies.tabular.segmentation import apply_segmentation
from ml.feature_freezing.freeze_strategies.tabular.validation import (
    validate_include_exclude_columns,
    validate_min_class_count,
    validate_min_rows,
)
from ml.feature_freezing.utils.operators import validate_operators
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class IngestionStep(PipelineStep[FreezeContext]):
    name = "ingestion"

    def before(self, ctx: FreezeContext) -> None:
        logger.debug("Starting data ingestion step.")

    def after(self, ctx: FreezeContext) -> None:
        logger.debug("Completed data ingestion step.")
    
    def run(self, ctx: FreezeContext) -> FreezeContext:
        data, loader_validation_hash = load_data_with_loader_validation_hash(
            ctx.config.data.path,
            ctx.config.data.format
        )

        data_metadata = load_json(ctx.config.data.metadata_path)
        validate_dataset(data_path=ctx.config.data.path, metadata=data_metadata)

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
        ctx.loader_validation_hash = loader_validation_hash

        return ctx