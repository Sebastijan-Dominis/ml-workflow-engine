import logging
logger = logging.getLogger(__name__)
from datetime import datetime
from pathlib import Path

from ml.feature_freezing.freeze_strategies.tabular.pipeline.ingestion import ingest_data
from ml.feature_freezing.freeze_strategies.tabular.pipeline.preprocessing import prepare_dataset
from ml.feature_freezing.freeze_strategies.tabular.pipeline.splitting import split_dataset
from ml.feature_freezing.freeze_strategies.tabular.pipeline.persistence import persist_all
from ml.feature_freezing.freeze_strategies.tabular.pipeline.metadata import build_metadata
from ml.feature_freezing.freeze_strategies.base import FreezeStrategy
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

# TODO: consider a runner with steps, instead of function-based pipeline
class FreezeTabular(FreezeStrategy):
    def freeze(self, config: TabularFeaturesConfig) -> tuple[Path, dict]:
        data, data_hash = ingest_data(config)

        X, y = prepare_dataset(data, config)

        splits = split_dataset(X, y, config)
        
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        snapshot_path, schema_path = persist_all(config, splits, now)

        config_hash = self.hash_config(config)

        metadata = build_metadata(
            config,
            snapshot_path,
            schema_path,
            splits,
            data_hash,
            config_hash
        )

        return snapshot_path, metadata