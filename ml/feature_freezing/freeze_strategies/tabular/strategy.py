"""Concrete feature-freezing strategy for tabular datasets."""

import logging

from ml.feature_freezing.constants.output import FreezeOutput
from ml.feature_freezing.freeze_strategies.base import FreezeStrategy
from ml.feature_freezing.freeze_strategies.config.validate_feature_registry import (
    validate_feature_registry,
)
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import FreezeContext
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.ingestion import IngestionStep
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.metadata import MetadataStep
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.persistence import PersistenceStep
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.preprocessing import (
    PreprocessingStep,
)
from ml.utils.pipeline_core.runner import PipelineRunner

logger = logging.getLogger(__name__)

class FreezeTabular(FreezeStrategy):
    """Run the tabular freeze pipeline and return snapshot metadata."""

    def freeze(
        self,
        config: TabularFeaturesConfig,
        *,
        snapshot_binding_key: str | None,
        timestamp: str,
        snapshot_id: str,
        start_time: float,
        owner: str
    ) -> FreezeOutput:
        """Execute tabular freezing steps using pipeline runner orchestration.

        Args:
            config: Validated tabular feature-freezing configuration.
            timestamp: Run timestamp string used for metadata and paths.
            snapshot_id: Unique snapshot identifier.
            start_time: Process start time used for runtime metadata.
            owner: Owner identifier stored in snapshot metadata.
            snapshot_binding_key: Optional key for a snapshot binding to define which snapshot to load for each dataset.
        Returns:
            Freeze output containing persisted snapshot path and metadata.
        """
        if not isinstance(config, TabularFeaturesConfig):
            config = validate_feature_registry(config.dict(), "tabular")
        ctx = FreezeContext(
            config=config,
            timestamp=timestamp,
            snapshot_id=snapshot_id,
            start_time=start_time,
            owner=owner,
            snapshot_binding_key=snapshot_binding_key
        )
        runner = PipelineRunner[FreezeContext](steps=[
            IngestionStep(),
            PreprocessingStep(),
            PersistenceStep(),
            MetadataStep(hash_config=self.hash_config)
        ])
        ctx = runner.run(ctx)
        output = FreezeOutput(snapshot_path=ctx.require_snapshot_path, metadata=ctx.require_metadata)
        return output
