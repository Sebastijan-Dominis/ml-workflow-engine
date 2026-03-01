import logging

from ml.feature_freezing.constants.output import FreezeOutput
from ml.feature_freezing.freeze_strategies.base import FreezeStrategy
from ml.feature_freezing.freeze_strategies.config.validate_feature_registry import \
    validate_feature_registry
from ml.feature_freezing.freeze_strategies.tabular.config.models import \
    TabularFeaturesConfig
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import \
    FreezeContext
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.ingestion import \
    IngestionStep
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.metadata import \
    MetadataStep
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.persistence import \
    PersistenceStep
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.preprocessing import \
    PreprocessingStep
from ml.utils.pipeline_core.runner import PipelineRunner

logger = logging.getLogger(__name__)

class FreezeTabular(FreezeStrategy):
    def freeze(
        self, 
        config: TabularFeaturesConfig, 
        *, 
        timestamp: str, 
        snapshot_id: str, 
        start_time: float, 
        owner: str
    ) -> FreezeOutput:
        if not isinstance(config, TabularFeaturesConfig):
            config = validate_feature_registry(config.dict(), "tabular")
        ctx = FreezeContext(
            config=config, 
            timestamp=timestamp, 
            snapshot_id=snapshot_id, 
            start_time=start_time, 
            owner=owner
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