import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config.validation_schemas.model_specs import SplitConfig, DATA_TYPE
from ml.registry.tabular_splits import TabularSplits

SPLIT = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]

def random_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int, stratify: pd.Series | None) -> SPLIT:
    return tuple(train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify))

def split_data(X: pd.DataFrame, y: pd.Series, split_cfg: SplitConfig, test_size: float) -> SPLIT:
    # Expandable for future split strategies
    SPLIT_REGISTRY = {
        "random": random_split,
    }

    split_func = SPLIT_REGISTRY[split_cfg.strategy]

    X1, X2, y1, y2 = split_func(
        X, y,
        test_size=test_size,
        random_state=split_cfg.random_state,
        stratify=y if split_cfg.stratify_by and isinstance(y, pd.Series) else None
    )

    return X1, X2, y1, y2

def get_splits_tabular(X: pd.DataFrame, y: pd.Series, split_cfg: SplitConfig) -> TabularSplits:
    X_train_val, X_test, y_train_val, y_test = split_data(
        X,
        y,
        split_cfg,
        test_size=split_cfg.test_size,
    )

    relative_val_size = (
        split_cfg.val_size /
        (1.0 - split_cfg.test_size)
    )

    X_train, X_val, y_train, y_val = split_data(
        X_train_val,
        y_train_val,
        split_cfg,
        test_size=relative_val_size,
    )

    splits = TabularSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    return splits

def get_splits(
    X: pd.DataFrame, 
    y: pd.Series,
    *, 
    split_cfg: SplitConfig, 
    data_type: DATA_TYPE
) -> TabularSplits:
    if data_type == "tabular":
        return get_splits_tabular(X, y, split_cfg)
    elif data_type == "time-series":
        raise NotImplementedError("Time-series split not implemented yet.")