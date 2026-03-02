import logging
from pathlib import Path

import pandas as pd

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)

def save_predictions(predictions_df: dict[str, pd.DataFrame], target_dir: Path) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    try:
        for split_name, df in predictions_df.items():
            split_path = target_dir / f"predictions_{split_name}.parquet"
            df.to_parquet(split_path)
            logger.info(f"Saved predictions for the {split_name} split to {split_path}")
            paths[split_name] = str(split_path)
        return paths
    
    except Exception as e:
        msg = f"Failed to save predictions to {target_dir}. "
        logger.error(msg + f"Details: {e}")
        raise PersistenceError(msg) from e