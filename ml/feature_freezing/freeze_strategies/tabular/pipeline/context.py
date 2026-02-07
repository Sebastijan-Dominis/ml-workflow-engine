import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import Optional

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.feature_freezing.freeze_strategies.tabular.pipeline.artifacts import TabularSplits
from ml.exceptions import RuntimeMLException

@dataclass
class FreezeContext:
    config: TabularFeaturesConfig

    timestamp: Optional[str] = None
    snapshot_id: Optional[str] = None

    data: Optional[pd.DataFrame] = None
    data_hash: Optional[str] = None

    X: Optional[pd.DataFrame] = None
    y: Optional[pd.DataFrame] = None

    splits: Optional[TabularSplits] = None

    snapshot_path: Optional[Path] = None
    schema_path: Optional[Path] = None

    metadata: Optional[dict] = None
    config_hash: Optional[str] = None

    @property
    def require_timestamp(self) -> str:
        if self.timestamp is None:
            msg = "Timestamp not set. Ensure that the timestamp is provided when calling freeze()."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.timestamp

    @property
    def require_snapshot_id(self) -> str:
        if self.snapshot_id is None:
            msg = "Snapshot ID not set. Ensure that the snapshot_id is provided when calling freeze()."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.snapshot_id

    @property
    def require_data(self) -> pd.DataFrame:
        if self.data is None:
            msg = "Data not loaded yet. Ensure that the ingestion step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.data

    @property
    def require_data_hash(self) -> str:
        if self.data_hash is None:
            msg = "Data hash not computed yet. Ensure that the ingestion step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.data_hash

    @property
    def require_X(self) -> pd.DataFrame:
        if self.X is None:
            msg = "Features not prepared yet. Ensure that the preprocessing step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.X

    @property
    def require_y(self) -> pd.DataFrame:
        if self.y is None:
            msg = "Target not prepared yet. Ensure that the preprocessing step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.y
    
    @property
    def require_splits(self) -> TabularSplits:
        if self.splits is None:
            msg = "Data not split yet. Ensure that the splitting step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.splits
    
    @property
    def require_snapshot_path(self) -> Path:
        if self.snapshot_path is None:
            msg = "Snapshot not persisted yet. Ensure that the persistence step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.snapshot_path
    
    @property
    def require_schema_path(self) -> Path:
        if self.schema_path is None:
            msg = "Schema not persisted yet. Ensure that the persistence step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.schema_path
    
    @property
    def require_config_hash(self) -> str:
        if self.config_hash is None:
            msg = "Config hash not computed yet. Ensure that the metadata step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.config_hash
    
    @property
    def require_metadata(self) -> dict:
        if self.metadata is None:
            msg = "Metadata not created yet. Ensure that the metadata step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.metadata