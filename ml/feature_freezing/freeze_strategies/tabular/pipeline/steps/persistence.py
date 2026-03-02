"""Persistence step for tabular feature-freezing pipeline."""

import logging

from ml.exceptions import PersistenceError
from ml.feature_freezing.freeze_strategies.tabular.persistence import (
    persist_feature_snapshot, save_derived_schema, save_input_schema)
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import \
    FreezeContext
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class PersistenceStep(PipelineStep[FreezeContext]):
    """Persist feature snapshot and associated schema artifacts."""

    name = "persistence"

    def before(self, ctx: FreezeContext) -> None:
        """Emit pre-step log message.

        Args:
            ctx: Freeze pipeline context.

        Returns:
            None: Emits logging side effect only.
        """
        logger.info("Starting data persistence step.")
        
    def after(self, ctx: FreezeContext) -> None:
        """Emit post-step log message.

        Args:
            ctx: Freeze pipeline context.

        Returns:
            None: Emits logging side effect only.
        """
        logger.info("Completed data persistence step.")

    def run(self, ctx: FreezeContext) -> FreezeContext:
        """Persist data and schemas, then update context artifact paths.

        Args:
            ctx: Freeze pipeline context.

        Returns:
            FreezeContext: Updated context with persisted artifact paths.
        """
        config = ctx.config

        features = ctx.require_features

        snapshot_path, data_path = persist_feature_snapshot(
            config,
            features=features,
            snapshot_id=ctx.snapshot_id,
        )

        schema_path = config.feature_store_path

        if "row_id" in features.columns:
            features_without_row_id = features.drop(columns=["row_id"])
        else:
            msg = "Expected 'row_id' column in features, but it was not found."
            logger.error(msg)
            raise PersistenceError(msg)
        try:
            save_input_schema(schema_path, features_without_row_id)
        except Exception as e:
            logger.exception("Failed to save input schema")
            raise PersistenceError(
                f"Could not save input schema at {schema_path}"
            ) from e

        try:
            if config.operators:
                save_derived_schema(
                    schema_path,
                    features=features_without_row_id,
                    operator_names=config.operators.names,
                    mode=config.operators.mode,
                )
        except Exception as e:
            logger.exception("Failed to save derived schema")
            raise PersistenceError(
                f"Could not save derived schema at {schema_path}"
            ) from e

        ctx.snapshot_path = snapshot_path
        ctx.schema_path = schema_path
        ctx.data_path = data_path

        return ctx
