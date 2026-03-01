import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.exceptions import RuntimeMLException
from ml.feature_freezing.freeze_strategies.tabular.config.models import \
    TabularFeaturesConfig
from ml.utils.data.models import DataLineageEntry

logger = logging.getLogger(__name__)
@dataclass
class FreezeContext:
    config: TabularFeaturesConfig
    timestamp: str
    snapshot_id: str
    start_time: float
    owner: str

    data: Optional[pd.DataFrame] = None
    data_lineage: Optional[list[DataLineageEntry]] = None

    features: Optional[pd.DataFrame] = None

    snapshot_path: Optional[Path] = None
    schema_path: Optional[Path] = None
    data_path: Optional[Path] = None

    metadata: Optional[dict] = None
    config_hash: Optional[str] = None

    @property
    def require_data(self) -> pd.DataFrame:
        if self.data is None:
            msg = "Data not loaded yet. Ensure that the ingestion step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.data

    @property
    def require_data_lineage(self) -> list[DataLineageEntry]:
        if self.data_lineage is None:
            msg = "Data lineage not computed yet. Ensure that the ingestion step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.data_lineage

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