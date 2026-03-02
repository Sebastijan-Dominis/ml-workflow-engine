"""CSV persistence helpers for explainability metric tables."""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

def save_metrics_csv(
    metrics: pd.DataFrame,
    *, 
    target_file: Path, 
    name: Literal["Feature importances", "SHAP importances"]
) -> None:
    """Persist explainability metric dataframe to CSV at target path.

    Args:
        metrics: Explainability metrics dataframe to persist.
        target_file: Destination CSV file path.
        name: Human-readable metric table name for logging.

    Returns:
        None.
    """

    target_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics.to_csv(target_file, index=False)
        logger.info(f'{name} successfully saved to {target_file}.')
    except Exception as e:
        msg = f"Failed to save {name} to {target_file}"
        logger.exception(msg)
        raise Exception(msg) from e