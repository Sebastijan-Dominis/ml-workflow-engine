import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.exceptions import RuntimeMLException

logger = logging.getLogger(__name__)
@dataclass
class SearchContext:
    model_cfg: SearchModelConfig

    X_train: Optional[pd.DataFrame] = None
    y_train: Optional[pd.DataFrame] = None
    feature_lineage: Optional[list[dict]] = None
    input_schema: Optional[pd.DataFrame] = None
    derived_schema: Optional[pd.DataFrame] = None
    pipeline_cfg: Optional[dict] = None
    pipeline_hash: Optional[str] = None
    cat_features: Optional[list[str]] = None

    best_params_1: Optional[dict] = None
    broad_result: Optional[dict] = None

    narrow_disabled: Optional[bool] = None
    best_params: Optional[dict] = None
    narrow_result: Optional[dict] = None

    @property
    def require_X_train(self) -> pd.DataFrame:
        if self.X_train is None:
            msg = "X_train not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.X_train
    
    @property
    def require_y_train(self) -> pd.DataFrame:
        if self.y_train is None:
            msg = "y_train not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.y_train

    @property
    def require_feature_lineage(self) -> list[dict]:
        if self.feature_lineage is None:
            msg = "Feature lineage not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.feature_lineage

    @property
    def require_input_schema(self) -> pd.DataFrame:
        if self.input_schema is None:
            msg = "Input schema not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.input_schema

    @property
    def require_derived_schema(self) -> pd.DataFrame:
        if self.derived_schema is None:
            msg = "Derived schema not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.derived_schema

    @property
    def require_pipeline_cfg(self) -> dict:
        if self.pipeline_cfg is None:
            msg = "Pipeline config not loaded yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.pipeline_cfg
    
    @property
    def require_pipeline_hash(self) -> str:
        if self.pipeline_hash is None:
            msg = "Pipeline hash not computed yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.pipeline_hash

    @property
    def require_cat_features(self) -> list[str]:
        if self.cat_features is None:
            msg = "Categorical features not prepared yet. Ensure that the preparation step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.cat_features
    
    @property
    def require_best_params_1(self) -> dict:
        if self.best_params_1 is None:
            msg = "Best parameters from broad search not available yet. Ensure that the broad search step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.best_params_1
    
    @property
    def require_broad_result(self) -> dict:
        if self.broad_result is None:
            msg = "Broad search result not available yet. Ensure that the broad search step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.broad_result
    
    @property
    def require_narrow_disabled(self) -> bool:
        if self.narrow_disabled is None:
            msg = "Narrow search enabled/disabled flag not set yet. Ensure that the broad search step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.narrow_disabled

    @property
    def require_best_params(self) -> dict:
        if self.best_params is None:
            msg = "Best parameters from narrow search not available yet. Ensure that the narrow search step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.best_params
    
    @property
    def require_narrow_result(self) -> dict:
        if self.narrow_result is None:
            msg = "Narrow search result not available yet. Ensure that the narrow search step has been run."
            logger.error(msg)
            raise RuntimeMLException(msg)
        return self.narrow_result