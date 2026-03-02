"""Typed containers for train/validation/test evaluation split payloads."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataSplits:
    """Feature/target tuples for train, validation, and test splits."""

    train: tuple[pd.DataFrame, pd.Series]
    val: tuple[pd.DataFrame, pd.Series]
    test: tuple[pd.DataFrame, pd.Series]