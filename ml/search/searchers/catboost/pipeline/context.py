"""Execution context model shared across CatBoost search pipeline steps."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.config.schemas.model_cfg import SearchModelConfig
from ml.exceptions import RuntimeMLException
from ml.types.splits import AllSplitsInfo
from ml.utils.experiments.class_weights.constants import \
    SUPPORTED_SCORING_FUNCTIONS

logger = logging.getLogger(__name__)

@dataclass
class SearchContext:
    """Mutable state container for CatBoost search pipeline orchestration."""

    model_cfg: SearchModelConfig
    strict: bool
    failure_management_dir: Path

    X_train: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    splits_info: Optional[AllSplitsInfo] = None
    feature_lineage: Optional[list[dict]] = None
    input_schema: Optional[pd.DataFrame] = None
    derived_schema: Optional[pd.DataFrame] = None
    pipeline_cfg: Optional[dict] = None
    pipeline_hash: Optional[str] = None
    cat_features: Optional[list[str]] = None
    scoring: Optional[SUPPORTED_SCORING_FUNCTIONS] = None
    class_weights: Optional[dict] = None

    best_params_1: Optional[dict] = None
    broad_result: Optional[dict] = None

    narrow_disabled: Optional[bool] = None
    best_params: Optional[dict] = None
    narrow_result: Optional[dict] = None

    @property
    def require_X_train(self) -> pd.DataFrame:
        """Return prepared training features or raise if unavailable.

        Returns:
            pd.DataFrame: Prepared training features.
        """

        if self.X_train is None:
            msg = "X_train not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.X_train
    
    @property
    def require_y_train(self) -> pd.Series:
        """Return prepared training target or raise if unavailable.

        Returns:
            pd.Series: Prepared training target.
        """

        if self.y_train is None:
            msg = "y_train not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.y_train

    @property
    def require_splits_info(self) -> AllSplitsInfo:
        """Return split summary metadata or raise if not prepared.

        Returns:
            AllSplitsInfo: Summary information for data splits.
        """

        if self.splits_info is None:
            msg = "Splits info not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.splits_info

    @property
    def require_feature_lineage(self) -> list[dict]:
        """Return feature lineage or raise if not prepared.

        Returns:
            list[dict]: Loaded feature lineage metadata.
        """

        if self.feature_lineage is None:
            msg = "Feature lineage not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.feature_lineage

    @property
    def require_input_schema(self) -> pd.DataFrame:
        """Return input schema dataframe or raise if not loaded.

        Returns:
            pd.DataFrame: Input feature schema.
        """

        if self.input_schema is None:
            msg = "Input schema not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.input_schema

    @property
    def require_derived_schema(self) -> pd.DataFrame:
        """Return derived schema dataframe or raise if not loaded.

        Returns:
            pd.DataFrame: Derived feature schema.
        """

        if self.derived_schema is None:
            msg = "Derived schema not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.derived_schema

    @property
    def require_pipeline_cfg(self) -> dict:
        """Return loaded pipeline config or raise if missing.

        Returns:
            dict: Loaded pipeline configuration.
        """

        if self.pipeline_cfg is None:
            msg = "Pipeline config not loaded yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.pipeline_cfg
    
    @property
    def require_pipeline_hash(self) -> str:
        """Return computed pipeline hash or raise if not set.

        Returns:
            str: Pipeline configuration hash.
        """

        if self.pipeline_hash is None:
            msg = "Pipeline hash not computed yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.pipeline_hash

    @property
    def require_cat_features(self) -> list[str]:
        """Return categorical feature list or raise if not prepared.

        Returns:
            list[str]: Categorical feature names.
        """

        if self.cat_features is None:
            msg = "Categorical features not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.cat_features
    
    @property
    def require_scoring(self) -> SUPPORTED_SCORING_FUNCTIONS:
        """Return resolved scoring function or raise if unset.

        Returns:
            SUPPORTED_SCORING_FUNCTIONS: Selected scoring metric.
        """

        if self.scoring is None:
            msg = "Scoring function not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.scoring

    @property
    def require_best_params_1(self) -> dict:
        """Return broad-phase best params or raise if broad step not run.

        Returns:
            dict: Best parameters from broad search.
        """

        if self.best_params_1 is None:
            msg = "Best parameters from broad search not available yet. Ensure that the broad search step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.best_params_1
    
    @property
    def require_broad_result(self) -> dict:
        """Return broad-phase search result payload or raise if missing.

        Returns:
            dict: Broad search result payload.
        """

        if self.broad_result is None:
            msg = "Broad search result not available yet. Ensure that the broad search step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.broad_result
    
    @property
    def require_narrow_disabled(self) -> bool:
        """Return narrow-enabled flag or raise if not decided yet.

        Returns:
            bool: Whether narrow search is disabled.
        """

        if self.narrow_disabled is None:
            msg = "Narrow search enabled/disabled flag not set yet. Ensure that the broad search step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.narrow_disabled

    @property
    def require_best_params(self) -> dict:
        """Return narrow-phase best params or raise if missing.

        Returns:
            dict: Best parameters from narrow search.
        """

        if self.best_params is None:
            msg = "Best parameters from narrow search not available yet. Ensure that the narrow search step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.best_params
    
    @property
    def require_narrow_result(self) -> dict:
        """Return narrow-phase result payload or raise if missing.

        Returns:
            dict: Narrow search result payload.
        """

        if self.narrow_result is None:
            msg = "Narrow search result not available yet. Ensure that the narrow search step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.narrow_result