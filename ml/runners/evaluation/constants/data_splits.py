from dataclasses import dataclass

import pandas as pd


@dataclass
class DataSplits:
    train: tuple[pd.DataFrame, pd.Series]
    val: tuple[pd.DataFrame, pd.Series]
    test: tuple[pd.DataFrame, pd.Series]