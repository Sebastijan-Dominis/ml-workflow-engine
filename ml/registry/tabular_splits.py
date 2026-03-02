"""Dataclass containers describing tabular dataset splits and split stats."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitInfo:
    """Summary information for a single dataset split."""

    n_rows: int
    class_distribution: dict | None = None
    positive_rate: float | None = None

@dataclass
class AllSplitsInfo:
    """Grouped split summaries for train/validation/test partitions."""

    train: SplitInfo
    val: SplitInfo
    test: SplitInfo

@dataclass
class TabularSplits:
    """Concrete train/validation/test feature and target split payloads."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series