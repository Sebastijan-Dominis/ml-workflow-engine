import pandas as pd

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.feature_freezing.freeze_strategies.tabular.features import add_arrival_datetime, prepare_features
from ml.feature_freezing.freeze_strategies.tabular.validation import validate_constraints, validate_data_types, validate_target
from ml.feature_freezing.freeze_strategies.tabular.features import apply_operators

def prepare_dataset(data: pd.DataFrame, config: TabularFeaturesConfig) -> tuple[pd.DataFrame, pd.DataFrame]:

    X, y = prepare_features(data, config)

    X = add_arrival_datetime(X)

    validate_data_types(X, config)
    validate_target(y, config)
    validate_constraints(X, config)

    if config.operators and config.operators.mode == "materialized":
        X = apply_operators(X, config.operators.list)

    y = y.to_frame() if isinstance(y, pd.Series) else y

    return X, y
