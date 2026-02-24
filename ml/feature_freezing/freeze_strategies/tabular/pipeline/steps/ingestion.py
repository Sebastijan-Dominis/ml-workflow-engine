import logging
from pathlib import Path

from ml.exceptions import DataError
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import \
    FreezeContext
from ml.feature_freezing.utils.operators import validate_operators
from ml.utils.data.loader import load_data_with_loader_validation_hash
from ml.utils.data.validate_dataset import validate_dataset
from ml.utils.data.validate_min_rows import validate_min_rows
from ml.utils.data.validate_row_id import validate_row_id
from ml.utils.loaders import load_json
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
            Path(ctx.config.data.path),
            ctx.config.data.format
        )

        validate_row_id(data)

        data_metadata = load_json(Path(ctx.config.data.metadata_path))
        validate_dataset(data_path=Path(ctx.config.data.path), metadata=data_metadata)

        validate_min_rows(data, ctx.config.min_rows)

        if ctx.config.operators:
            validate_operators(
                ctx.config.operators.names,
                ctx.config.operators.hash
            )

        ctx.data = data
        ctx.loader_validation_hash = loader_validation_hash

        return ctx