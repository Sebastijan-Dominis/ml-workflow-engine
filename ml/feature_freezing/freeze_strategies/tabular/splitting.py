import pandas as pd
from sklearn.model_selection import train_test_split
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

def random_split(X: pd.DataFrame, y: pd.DataFrame, test_size: float, random_state: int, stratify: pd.Series | None) -> list[pd.DataFrame]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def split_data(X: pd.DataFrame, y: pd.DataFrame, config: TabularFeaturesConfig, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Expandable for future split strategies
    SPLIT_REGISTRY = {
        "random": random_split,
    }

    split_func = SPLIT_REGISTRY[config.split.strategy]

    X1, X2, y1, y2 = split_func(
        X, y,
        test_size=test_size,
        random_state=config.split.random_state,
        stratify=y if config.split.stratify_by and isinstance(y, pd.Series) else None
    )

    X1 = X1.to_frame() if isinstance(X1, pd.Series) else X1
    X2 = X2.to_frame() if isinstance(X2, pd.Series) else X2
    y1 = y1.to_frame() if isinstance(y1, pd.Series) else y1
    y2 = y2.to_frame() if isinstance(y2, pd.Series) else y2

    return X1, X2, y1, y2