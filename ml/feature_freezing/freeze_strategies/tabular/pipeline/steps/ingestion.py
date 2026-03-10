"""Ingestion step for tabular feature-freezing pipeline."""

import logging

from ml.data.validation.validate_min_rows import validate_min_rows
from ml.data.validation.validate_row_id import validate_row_id
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import FreezeContext
from ml.feature_freezing.utils.data_loader import load_data_with_lineage
from ml.features.validation.validate_operators import validate_operators
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class IngestionStep(PipelineStep[FreezeContext]):
    """Load source data, validate prerequisites, and attach lineage."""

    name = "ingestion"

    def before(self, ctx: FreezeContext) -> None:
        """Emit pre-step log message.

        Args:
            ctx: Freeze pipeline context.

        Returns:
            None: Emits logging side effect only.
        """
        logger.info("Starting data ingestion step.")

    def after(self, ctx: FreezeContext) -> None:
        """Emit post-step log message.

        Args:
            ctx: Freeze pipeline context.

        Returns:
            None: Emits logging side effect only.
        """
        logger.info("Completed data ingestion step.")

    def run(self, ctx: FreezeContext) -> FreezeContext:
        """Execute ingestion workflow and update context with data and lineage.

        Args:
            ctx: Freeze pipeline context.

        Returns:
            FreezeContext: Updated context with loaded data and lineage.
        """
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
