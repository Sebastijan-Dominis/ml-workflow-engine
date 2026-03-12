"""Execution context model for tabular feature-freezing pipeline stages."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.exceptions import RuntimeMLError
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.types import DataLineageEntry

logger = logging.getLogger(__name__)

@dataclass
class FreezeContext:
    """Mutable context shared across tabular freeze pipeline steps."""

    config: TabularFeaturesConfig
    timestamp: str
    snapshot_id: str
    start_time: float
    owner: str

    data: pd.DataFrame | None = None
    data_lineage: list[DataLineageEntry] | None = None

    features: pd.DataFrame | None = None

    snapshot_path: Path | None = None
    schema_path: Path | None = None
    data_path: Path | None = None

    metadata: dict | None = None
    config_hash: str | None = None

    @property
    def require_data(self) -> pd.DataFrame:
        """Return loaded source data or raise if ingestion has not run.

        Returns:
            pd.DataFrame: Loaded source data.
        """
        if self.data is None:
            msg = "Data not loaded yet. Ensure that the ingestion step has been run."
            logger.error(msg)
            raise RuntimeMLError(msg)
        return self.data

    @property
    def require_data_lineage(self) -> list[DataLineageEntry]:
        """Return data lineage entries or raise if not yet populated.

        Returns:
            list[DataLineageEntry]: Source data lineage entries.
        """
        if self.data_lineage is None:
            msg = "Data lineage not computed yet. Ensure that the ingestion step has been run."
            logger.error(msg)
            raise RuntimeMLError(msg)
        return self.data_lineage

    @property
    def require_features(self) -> pd.DataFrame:
        """Return prepared feature dataframe or raise if unavailable.

        Returns:
            pd.DataFrame: Prepared feature dataframe.
        """
        if self.features is None:
            msg = "Features not prepared yet. Ensure that the preprocessing step has been run."
            logger.error(msg)
            raise RuntimeMLError(msg)
        return self.features

    @property
    def require_snapshot_path(self) -> Path:
        """Return persisted snapshot path or raise if persistence not run.

        Returns:
            Path: Snapshot directory path.
        """
        if self.snapshot_path is None:
            msg = "Snapshot not persisted yet. Ensure that the persistence step has been run."
            logger.error(msg)
            raise RuntimeMLError(msg)
        return self.snapshot_path

    @property
    def require_schema_path(self) -> Path:
        """Return schema path or raise if persistence not run.

        Returns:
            Path: Persisted input schema path.
        """
        if self.schema_path is None:
            msg = "Schema not persisted yet. Ensure that the persistence step has been run."
            logger.error(msg)
            raise RuntimeMLError(msg)
        return self.schema_path

    @property
    def require_data_path(self) -> Path:
        """Return persisted feature data path or raise if unset.

        Returns:
            Path: Persisted features data-file path.
        """
        if self.data_path is None:
            msg = "Data path not set. Ensure that the persistence step has been run and data path is set in the context."
            logger.error(msg)
            raise RuntimeMLError(msg)
        return self.data_path

    @property
    def require_config_hash(self) -> str:
        """Return computed config hash or raise if metadata step not run.

        Returns:
            str: Computed feature-freezing config hash.
        """
        if self.config_hash is None:
            msg = "Config hash not computed yet. Ensure that the metadata step has been run."
            logger.error(msg)
            raise RuntimeMLError(msg)
        return self.config_hash

    @property
    def require_metadata(self) -> dict:
        """Return assembled metadata payload or raise if missing.

        Returns:
            dict: Assembled metadata payload.
        """
        if self.metadata is None:
            msg = "Metadata not created yet. Ensure that the metadata step has been run."
            logger.error(msg)
            raise RuntimeMLError(msg)
        return self.metadata
