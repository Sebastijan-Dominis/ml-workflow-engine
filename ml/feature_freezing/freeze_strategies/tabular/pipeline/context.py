import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.exceptions import RuntimeMLException
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

logger = logging.getLogger(__name__)
@dataclass
class FreezeContext:
    config: TabularFeaturesConfig

    timestamp: Optional[str] = None
    snapshot_id: Optional[str] = None
    start_time: Optional[float] = None

    data: Optional[pd.DataFrame] = None
    loader_validation_hash: Optional[str] = None

    features: Optional[pd.DataFrame] = None

    snapshot_path: Optional[Path] = None
    schema_path: Optional[Path] = None
    data_path: Optional[Path] = None

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
    def require_start_time(self) -> float:
        if self.start_time is None:
            msg = "Start time not set. Ensure that the start_time is provided when calling freeze()."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.start_time

    @property
    def require_data(self) -> pd.DataFrame:
        if self.data is None:
            msg = "Data not loaded yet. Ensure that the ingestion step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.data

    @property
    def require_loader_validation_hash(self) -> str:
        if self.loader_validation_hash is None:
            msg = "Data hash not computed yet. Ensure that the ingestion step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.loader_validation_hash

    @property
    def require_features(self) -> pd.DataFrame:
        if self.features is None:
            msg = "Features not prepared yet. Ensure that the preprocessing step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.features
    
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
    def require_data_path(self) -> Path:
        if self.data_path is None:
            msg = "Data path not set. Ensure that the persistence step has been run and data path is set in the context."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.data_path
    
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