from dataclasses import dataclass

import pandas as pd


@dataclass
class EVALUATE_OUTPUT:
    metrics: dict[str, dict[str, float]]
    prediction_dfs: dict[str, pd.DataFrame]
    lineage: list[dict]