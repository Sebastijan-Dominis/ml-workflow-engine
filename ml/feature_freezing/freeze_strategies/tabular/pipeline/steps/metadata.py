import logging
logger = logging.getLogger(__name__)
from pathlib import Path

from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import FreezeContext
from ml.utils.pipeline_core.step import PipelineStep
from ml.feature_freezing.freeze_strategies.tabular.persistence import create_metadata
from ml.feature_freezing.freeze_strategies.tabular.io import hash_feature_set, validate_feature_set_hashes_match
from ml.utils.git import get_git_commit
from ml.feature_freezing.utils.schema import hash_data_schema

class MetadataStep(PipelineStep[FreezeContext]):
    name = "metadata"

    def before(self, ctx: FreezeContext) -> None:
        logger.debug("Starting metadata step.")

    def after(self, ctx: FreezeContext) -> None:
        logger.debug("Completed metadata step.")

    def __init__(self, hash_config):
        self.hash_config = hash_config

    def run(self, ctx: FreezeContext) -> FreezeContext:
        ctx.config_hash = self.hash_config(ctx.config)

        splits = ctx.require_splits
        X_train = splits.X_train
        X_val = splits.X_val
        X_test = splits.X_test
        y_train = splits.y_train
        y_val = splits.y_val
        y_test = splits.y_test

        train_schema_hash = hash_data_schema(X_train)
        val_schema_hash = hash_data_schema(X_val)
        test_schema_hash = hash_data_schema(X_test)

        feature_set_hash = hash_feature_set(X_train)

        validate_feature_set_hashes_match(X_val, feature_set_hash)
        validate_feature_set_hashes_match(X_test, feature_set_hash)

        git_commit = get_git_commit(Path("."))

        operators_hash = (
            ctx.config.operators.hash
            if ctx.config.operators else "none"
        )

        metadata = create_metadata(
            ctx.require_snapshot_path,
            ctx.require_schema_path,
            ctx.require_data_hash,
            train_schema_hash,
            val_schema_hash,
            test_schema_hash,
            operators_hash,
            ctx.require_config_hash,
            feature_set_hash,
            git_commit,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            ctx.config.target.problem_type,
        )

        ctx.metadata = metadata
        
        return ctx