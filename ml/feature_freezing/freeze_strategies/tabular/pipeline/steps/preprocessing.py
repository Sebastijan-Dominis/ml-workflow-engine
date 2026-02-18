import logging

import pandas as pd

from ml.feature_freezing.freeze_strategies.tabular.features import (
    apply_operators,
    prepare_features,
)
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import FreezeContext
from ml.feature_freezing.freeze_strategies.tabular.validation import (
    validate_constraints,
    validate_data_types,
    validate_target,
)
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class PreprocessingStep(PipelineStep[FreezeContext]):
    name = "preprocessing"

    def before(self, ctx: FreezeContext) -> None:
        logger.debug("Starting data preprocessing step.")
        
    def after(self, ctx: FreezeContext) -> None:
        logger.debug("Completed data preprocessing step.")

    def run(self, ctx: FreezeContext) -> FreezeContext:
        X, y = prepare_features(ctx.require_data, ctx.config)

        validate_data_types(X, ctx.config)
        validate_target(y, ctx.config)
        validate_constraints(X, ctx.config)

        if ctx.config.operators and ctx.config.operators.mode == "materialized":
            X = apply_operators(X, ctx.config.operators.list)
        y = y.to_frame() if isinstance(y, pd.Series) else y

        ctx.X = X
        ctx.y = y

        return ctx