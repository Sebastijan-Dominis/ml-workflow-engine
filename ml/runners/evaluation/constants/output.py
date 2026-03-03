"""Typed output model for evaluation runner results."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class EvaluateOutput:
    """Evaluation metrics, prediction artifacts, and feature lineage payload."""

    metrics: dict[str, dict[str, float]]
    prediction_dfs: dict[str, pd.DataFrame]
    lineage: list[dict]