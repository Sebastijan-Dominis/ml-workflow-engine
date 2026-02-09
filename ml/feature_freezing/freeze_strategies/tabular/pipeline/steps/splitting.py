import logging

from ml.feature_freezing.freeze_strategies.tabular.pipeline.artifacts import TabularSplits
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import FreezeContext
from ml.feature_freezing.freeze_strategies.tabular.splitting import split_data
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class SplittingStep(PipelineStep[FreezeContext]):
    name = "splitting"

    def before(self, ctx: FreezeContext) -> None:
        logger.debug("Starting data splitting step.")

    def after(self, ctx: FreezeContext) -> None:
        logger.debug("Completed data splitting step.")

    def run(self, ctx: FreezeContext) -> FreezeContext:
        X_train_val, X_test, y_train_val, y_test = split_data(
            ctx.require_X,
            ctx.require_y,
            ctx.config,
            test_size=ctx.config.split.test_size,
        )

        relative_val_size = (
            ctx.config.split.val_size /
            (1.0 - ctx.config.split.test_size)
        )

        X_train, X_val, y_train, y_val = split_data(
            X_train_val,
            y_train_val,
            ctx.config,
            test_size=relative_val_size,
        )

        ctx.splits = TabularSplits(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

        return ctx