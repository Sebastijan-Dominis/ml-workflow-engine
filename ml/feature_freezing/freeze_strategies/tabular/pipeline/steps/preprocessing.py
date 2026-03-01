import logging

from ml.feature_freezing.freeze_strategies.tabular.features import (
    apply_operators, prepare_features)
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import \
    FreezeContext
from ml.feature_freezing.freeze_strategies.tabular.validation import (
    validate_constraints, validate_data_types)
from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

class PreprocessingStep(PipelineStep[FreezeContext]):
    name = "preprocessing"

    def before(self, ctx: FreezeContext) -> None:
        logger.debug("Starting data preprocessing step.")
        
    def after(self, ctx: FreezeContext) -> None:
        logger.debug("Completed data preprocessing step.")

    def run(self, ctx: FreezeContext) -> FreezeContext:
        features = prepare_features(ctx.require_data, ctx.config)

        validate_data_types(features, ctx.config)
        validate_constraints(features, ctx.config)

        if ctx.config.operators and ctx.config.operators.mode == "materialized":
            features = apply_operators(features, ctx.config.operators.names, ctx.config.operators.required_features)

        ctx.features = features

        return ctx