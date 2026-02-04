import logging
logger = logging.getLogger(__name__)
import pandas as pd

from ml.exceptions import UserError
from ml.registry.feature_operators import FEATURE_OPERATORS
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

def prepare_features(data: pd.DataFrame, config: TabularFeaturesConfig) -> tuple[pd.DataFrame, pd.Series]:
    TARGET = config.target.name
    INCLUDE = config.columns.include
    
    # NOTE: `include` is the source of truth.
    # `exclude` exists only for validation and safety.
    X = data[INCLUDE].copy()
    y = data[TARGET].copy()

    return X, y

def add_arrival_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not all(col in df.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
        logger.warning("Arrival date columns not found, skipping arrival_datetime creation.")
        return df
    
    df['arrival_datetime'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'].astype(str) + '-' +
        df['arrival_date_day_of_month'].astype(str)
    )
    return df

def apply_operators(X: pd.DataFrame, operator_names: list[str]) -> pd.DataFrame:
    operators = []
    for name in operator_names:
        if name not in FEATURE_OPERATORS:
            raise UserError(f"Unknown operator: {name}")
        operators.append(FEATURE_OPERATORS[name]())
    for op in operators:
        X = op.transform(X)
    return X