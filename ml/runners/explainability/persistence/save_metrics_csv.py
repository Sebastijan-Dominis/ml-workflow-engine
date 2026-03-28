"""CSV persistence helpers for explainability metric tables."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd

from ml.exceptions import PersistenceError

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
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_file.parent,
            prefix="explainability.",
            suffix=".csv.tmp",
            delete=False,
        ) as tmp_file:
            temp_path = tmp_file.name
            metrics.to_csv(Path(temp_path), index=False)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(temp_path, target_file)

        msg = f"{name} successfully saved to {target_file}."
        logger.info(msg)
        print(msg)
    except Exception as e:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except OSError:
                logger.warning("Failed to clean up temporary explainability CSV file: %s", temp_path)

        msg = f"Failed to save {name} to {target_file}"
        logger.exception(msg)
        raise PersistenceError(msg) from e
