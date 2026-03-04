"""Prediction artifact persistence helpers for evaluation splits."""

import logging
from pathlib import Path

from ml.exceptions import PersistenceError
from ml.runners.evaluation.models.predictions import PredictionArtifacts, PredictionsPaths

logger = logging.getLogger(__name__)

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
        prediction_dfs.train.to_parquet(train_path)
        logger.info(f"Saved predictions for the train split to {train_path}")
        paths_raw["train_predictions_path"] = str(train_path)

        val_path = target_dir / "predictions_val.parquet"
        prediction_dfs.val.to_parquet(val_path)
        logger.info(f"Saved predictions for the validation split to {val_path}")
        paths_raw["val_predictions_path"] = str(val_path)

        test_path = target_dir / "predictions_test.parquet"
        prediction_dfs.test.to_parquet(test_path)
        logger.info(f"Saved predictions for the test split to {test_path}")
        paths_raw["test_predictions_path"] = str(test_path)

        paths = PredictionsPaths(**paths_raw)

        return paths

    except Exception as e:
        msg = f"Failed to save predictions to {target_dir}. "
        logger.error(msg + f"Details: {e}")
        raise PersistenceError(msg) from e
