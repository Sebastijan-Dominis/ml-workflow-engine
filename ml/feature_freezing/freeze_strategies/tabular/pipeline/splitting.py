import pandas as pd

from ml.feature_freezing.freeze_strategies.tabular.splitting import split_data
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

def split_dataset(X: pd.DataFrame, y: pd.DataFrame, config: TabularFeaturesConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    X_train_val, X_test, y_train_val, y_test = split_data(
        X,
        y,
        config,
        test_size=config.split.test_size,
    )

    relative_val_size = (
        config.split.val_size /
        (1.0 - config.split.test_size)
    )

    X_train, X_val, y_train, y_val = split_data(
        X_train_val,
        y_train_val,
        config,
        test_size=relative_val_size,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
