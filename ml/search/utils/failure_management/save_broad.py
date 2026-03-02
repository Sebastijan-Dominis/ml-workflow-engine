"""Persistence helper for saving broad-search resumable state."""

import json
import logging
from pathlib import Path

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)


def save_broad(
    *, 
    broad_result: dict, 
    best_params_1: dict, 
    tgt_file: Path
) -> None:
    """Persist broad search result and best params to JSON marker file."""

    broad_info = {
        "broad_result": broad_result,
        "best_params_1": best_params_1
    }

    try:
        with tgt_file.open("w", encoding="utf-8") as f:
            json.dump(broad_info, f, indent=2)
        logger.info(f"Best params successfully saved to {tgt_file}.")
    except Exception as e:
        msg = f"Failed to save best broad params to {tgt_file}"
        logger.exception(msg)
        raise PersistenceError(msg) from e