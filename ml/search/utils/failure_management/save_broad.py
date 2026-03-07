"""Persistence helper for saving broad-search resumable state."""

import json
import logging
import os
import tempfile
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

    tgt_file.parent.mkdir(parents=True, exist_ok=True)

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=tgt_file.parent,
            prefix="broad.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            temp_path = tmp_file.name
            json.dump(broad_info, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(temp_path, tgt_file)
        logger.info(f"Best params successfully saved to {tgt_file}.")
    except Exception as e:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except OSError:
                logger.warning("Failed to clean up temporary broad marker file: %s", temp_path)

        msg = f"Failed to save best broad params to {tgt_file}"
        logger.exception(msg)
        raise PersistenceError(msg) from e
