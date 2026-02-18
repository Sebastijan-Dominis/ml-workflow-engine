from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class ExplainabilityMetrics:
    top_k_feature_importances: Optional[pd.DataFrame] = None
    top_k_shap_importances: Optional[pd.DataFrame] = None