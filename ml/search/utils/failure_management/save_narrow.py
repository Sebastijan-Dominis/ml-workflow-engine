"""Persistence helper for saving narrow-search resumable state."""

import json
import logging
import os
import tempfile
from pathlib import Path

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)


def save_narrow(
    *,
    narrow_result: dict,
    best_params: dict,
    tgt_file: Path
) -> None:
    """Persist narrow search result and best params to JSON marker file."""

    narrow_info = {
        "narrow_result": narrow_result,
        "best_params": best_params
    }

    tgt_file.parent.mkdir(parents=True, exist_ok=True)

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=tgt_file.parent,
            prefix="narrow.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            temp_path = tmp_file.name
            json.dump(narrow_info, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(temp_path, tgt_file)
        logger.info(f"Narrow search done marker written to {tgt_file}.")
    except Exception as e:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except OSError:
                logger.warning("Failed to clean up temporary narrow marker file: %s", temp_path)

        msg = f"Failed to save narrow search done marker to {tgt_file}"
        logger.exception(msg)
        raise PersistenceError(msg) from e
