from datetime import datetime
import logging
logger = logging.getLogger(__name__)

from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import FreezeContext
from ml.utils.pipeline_core.step import PipelineStep
from ml.exceptions import PersistenceError
from ml.feature_freezing.freeze_strategies.tabular.persistence import persist_feature_snapshot
from ml.feature_freezing.freeze_strategies.tabular.persistence import save_input_schema, save_derived_schema

class PersistenceStep(PipelineStep[FreezeContext]):
    name = "persistence"

    def before(self, ctx: FreezeContext) -> None:
        logger.debug("Starting data persistence step.")
        
    def after(self, ctx: FreezeContext) -> None:
        logger.debug("Completed data persistence step.")

    def run(self, ctx: FreezeContext) -> FreezeContext:
        config = ctx.config
        splits = ctx.require_splits

        X_train = splits.X_train
        X_val = splits.X_val
        X_test = splits.X_test
        y_train = splits.y_train
        y_val = splits.y_val
        y_test = splits.y_test

        now = ctx.snapshot_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        snapshot_path = persist_feature_snapshot(
            config,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            now,
        )

        schema_path = config.feature_store_path

        try:
            save_input_schema(schema_path, X_train)
        except Exception as e:
            logger.exception("Failed to save input schema")
            raise PersistenceError(
                f"Could not save input schema at {schema_path}"
            ) from e

        try:
            if config.operators:
                save_derived_schema(
                    schema_path,
                    X_train,
                    config.operators.list,
                    config.operators.mode,
                )
        except Exception as e:
            logger.exception("Failed to save derived schema")
            raise PersistenceError(
                f"Could not save derived schema at {schema_path}"
            ) from e

        ctx.snapshot_path = snapshot_path
        ctx.schema_path = schema_path

        return ctx
