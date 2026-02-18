import logging

import pandas as pd

from ml.exceptions import UserError
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.registry.feature_operators import FEATURE_OPERATORS

logger = logging.getLogger(__name__)

def prepare_features(data: pd.DataFrame, config: TabularFeaturesConfig) -> tuple[pd.DataFrame, pd.Series]:
    TARGET = config.target.name
    INCLUDE = config.columns.include
    
    # NOTE: `include` is the source of truth.
    # `exclude` exists only for validation and safety.
    X = data[INCLUDE].copy()
    y = data[TARGET].copy()

    return X, y

def apply_operators(X: pd.DataFrame, operator_names: list[str]) -> pd.DataFrame:
    operators = []
    for name in operator_names:
        if name not in FEATURE_OPERATORS:
            raise UserError(f"Unknown operator: {name}")
        operators.append(FEATURE_OPERATORS[name]())
    for op in operators:
        X = op.transform(X)
    return X