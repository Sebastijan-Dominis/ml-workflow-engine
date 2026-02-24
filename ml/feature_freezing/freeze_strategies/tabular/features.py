import logging

import pandas as pd

from ml.exceptions import UserError, DataError
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.registry.feature_operators import FEATURE_OPERATORS

logger = logging.getLogger(__name__)

def prepare_features(data: pd.DataFrame, config: TabularFeaturesConfig) -> pd.DataFrame:
    INCLUDE = config.columns

    # include row_id as well; fail if not present, as it's required for downstream steps and evaluation
    if "row_id" not in data.columns:
        msg = "Input data must contain 'row_id' column."
        logger.error(msg)
        raise DataError(msg)
    missing_cols = set(INCLUDE) - set(data.columns)
    if missing_cols:
        msg = f"Missing required columns in input data: {missing_cols}"
        logger.error(msg)
        raise DataError(msg)
    X = data[["row_id"] + INCLUDE].copy()

    return X

def apply_operators(X: pd.DataFrame, operator_names: list[str], required_features: dict[str, list[str]]) -> pd.DataFrame:
    missing = set()
    for name, features in required_features.items():
        missing.update(set(features) - set(X.columns))
    if missing:
        msg = f"Missing required features for operators: {missing}"
        logger.error(msg)
        raise DataError(msg)

    operators = []
    for name in operator_names:
        if name not in FEATURE_OPERATORS:
            raise UserError(f"Unknown operator: {name}")
        operators.append(FEATURE_OPERATORS[name]())
    for op in operators:
        X = op.transform(X)
    return X