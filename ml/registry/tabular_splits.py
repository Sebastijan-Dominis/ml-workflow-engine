from dataclasses import dataclass

import pandas as pd

@dataclass
class SplitInfo:
    n_rows: int
    class_distribution: dict | None = None
    positive_rate: float | None = None

@dataclass
class AllSplitsInfo:
    train: SplitInfo
    val: SplitInfo
    test: SplitInfo

@dataclass
class TabularSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series