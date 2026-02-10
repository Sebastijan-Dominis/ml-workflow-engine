from typing import Protocol

import pandas as pd
import numpy as np

class ProbabilisticClassifier(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
