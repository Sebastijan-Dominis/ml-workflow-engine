"""Metadata step for tabular feature-freezing pipeline."""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from ml.feature_freezing.freeze_strategies.tabular.persistence import \
    create_metadata
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import \
    FreezeContext
from ml.feature_freezing.persistence.get_deps import get_deps
from ml.registry.hash_registry import hash_file
from ml.utils.features.hashing.hash_dataframe_content import \
    hash_dataframe_content
from ml.utils.features.hashing.hash_feature_schema import hash_feature_schema
from ml.utils.git import get_git_commit
from ml.utils.pipeline_core.step import PipelineStep
from ml.utils.runtime.runtime_info import get_runtime_info

logger = logging.getLogger(__name__)

class MetadataStep(PipelineStep[FreezeContext]):
    """Compute hashes/runtime metadata and attach final metadata payload."""

    name = "metadata"

    def before(self, ctx: FreezeContext) -> None:
        """Emit pre-step log message.

        Args:
            ctx: Freeze pipeline context.

        Returns:
            None: Emits logging side effect only.
        """

        logger.info("Starting metadata step.")

    def after(self, ctx: FreezeContext) -> None:
        """Emit post-step log message.

        Args:
            ctx: Freeze pipeline context.

        Returns:
            None: Emits logging side effect only.
        """

        logger.info("Completed metadata step.")

    def __init__(self, hash_config):
        """Initialize metadata step with injected config-hash function.

        Args:
            hash_config: Callable used to hash configuration payloads.

        Returns:
            None: Initializes step dependencies.
        """

        self.hash_config = hash_config

    def run(self, ctx: FreezeContext) -> FreezeContext:
        """Build and store metadata payload for the frozen snapshot.

        Args:
            ctx: Freeze pipeline context.

        Returns:
            FreezeContext: Updated context with computed metadata.
        """
        ctx.config_hash = self.hash_config(ctx.config)

        features = ctx.require_features
        in_memory_hash = hash_dataframe_content(features)

        feature_schema_hash = hash_feature_schema(features)

        file_hash = hash_file(ctx.require_data_path)

        git_commit = get_git_commit(Path("."))
        runtime_info = get_runtime_info()
        deps = get_deps()

        runtime = {
            "git_commit": git_commit,
            "runtime_info": runtime_info,
            "deps": deps,
            "python_executable": sys.executable
        }

        operators_hash = (
            ctx.config.operators.hash
            if ctx.config.operators else "none"
        )

        duration = round(time.perf_counter() - ctx.start_time, 3)
        
        metadata = create_metadata(
            timestamp = ctx.timestamp,
            snapshot_path = ctx.require_snapshot_path,
            schema_path = ctx.require_schema_path,
            data_lineage = [e.__dict__ for e in ctx.require_data_lineage],
            in_memory_hash = in_memory_hash,
            file_hash = file_hash,
            operators_hash = operators_hash,
            config_hash = ctx.require_config_hash,
            feature_schema_hash = feature_schema_hash,
            runtime = runtime,
            features = features,
            duration = duration,
            owner = ctx.owner
        )

        ctx.metadata = metadata
        
        return ctx