"""Prediction artifact persistence helpers for evaluation splits."""

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

from ml.exceptions import PersistenceError
from ml.runners.evaluation.models.predictions import PredictionArtifacts, PredictionsPaths

logger = logging.getLogger(__name__)


def _write_parquet_atomic(df: pd.DataFrame, target_path: Path) -> None:
    """Write parquet artifact atomically via temporary file and replace."""
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_path.parent,
            prefix=f"{target_path.stem}.",
            suffix=".parquet.tmp",
            delete=False,
        ) as tmp_file:
            temp_path = tmp_file.name

        df.to_parquet(temp_path)

        with open(temp_path, "r+b") as written_file:
            os.fsync(written_file.fileno())

        os.replace(temp_path, target_path)
    except Exception:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except OSError:
                logger.warning("Failed to clean up temporary predictions file: %s", temp_path)
        raise

def save_predictions(prediction_dfs: PredictionArtifacts, target_dir: Path) -> PredictionsPaths:
    """Persist per-split prediction dataframes as parquet artifacts.

    Args:
        prediction_dfs: Mapping from split names to prediction dataframes.
        target_dir: Directory where prediction parquet files are written.

    Returns:
        PredictionsPaths object with paths to the saved prediction artifacts.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    paths_raw = {}

    try:
        train_path = target_dir / "predictions_train.parquet"
        _write_parquet_atomic(prediction_dfs.train, train_path)
        logger.info(f"Saved predictions for the train split to {train_path}")
        paths_raw["train_predictions_path"] = str(train_path)

        val_path = target_dir / "predictions_val.parquet"
        _write_parquet_atomic(prediction_dfs.val, val_path)
        logger.info(f"Saved predictions for the validation split to {val_path}")
        paths_raw["val_predictions_path"] = str(val_path)

        test_path = target_dir / "predictions_test.parquet"
        _write_parquet_atomic(prediction_dfs.test, test_path)
        logger.info(f"Saved predictions for the test split to {test_path}")
        paths_raw["test_predictions_path"] = str(test_path)

        paths = PredictionsPaths(**paths_raw)

        return paths

    except Exception as e:
        msg = f"Failed to save predictions to {target_dir}. "
        logger.error(msg + f"Details: {e}")
        raise PersistenceError(msg) from e
