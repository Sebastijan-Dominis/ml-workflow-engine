import logging
from pathlib import Path

from ml.exceptions import DataError
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import \
    FreezeContext
from ml.feature_freezing.utils.operators import validate_operators
from ml.feature_freezing.utils.data_loader import load_data_with_lineage
from ml.utils.data.validate_min_rows import validate_min_rows
from ml.utils.data.validate_row_id import validate_row_id
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class IngestionStep(PipelineStep[FreezeContext]):
    name = "ingestion"

    def before(self, ctx: FreezeContext) -> None:
        logger.debug("Starting data ingestion step.")

    def after(self, ctx: FreezeContext) -> None:
        logger.debug("Completed data ingestion step.")
    
    def run(self, ctx: FreezeContext) -> FreezeContext:
        data, data_lineage = load_data_with_lineage(ctx.config)

        validate_row_id(data)

        validate_min_rows(data, ctx.config.min_rows)

        if ctx.config.operators:
            validate_operators(
                ctx.config.operators.names,
                ctx.config.operators.hash
            )

        ctx.data = data
        ctx.data_lineage = data_lineage

        return ctx