import logging
logger = logging.getLogger(__name__)
from pathlib import Path

from ml.feature_freezing.freeze_strategies.config.validate_feature_registry import validate_feature_registry
from ml.feature_freezing.freeze_strategies.tabular.pipeline.context import FreezeContext
from ml.feature_freezing.freeze_strategies.pipeline_core.runner import PipelineRunner
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.ingestion import IngestionStep
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.preprocessing import PreprocessingStep
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.splitting import SplittingStep
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.persistence import PersistenceStep
from ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.metadata import MetadataStep
from ml.feature_freezing.freeze_strategies.base import FreezeStrategy
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

class FreezeTabular(FreezeStrategy):
    def freeze(self, config: TabularFeaturesConfig, *, snapshot_id: str | None = None) -> tuple[Path, dict]:
        if not isinstance(config, TabularFeaturesConfig):
            config = validate_feature_registry(config.dict(), "tabular")
        ctx = FreezeContext(config=config, snapshot_id=snapshot_id)
        runner = PipelineRunner[FreezeContext](steps=[
            IngestionStep(),
            PreprocessingStep(),
            SplittingStep(),
            PersistenceStep(),
            MetadataStep(hash_config=self.hash_config)
        ])
        ctx = runner.run(ctx)
        return ctx.require_snapshot_path, ctx.require_metadata