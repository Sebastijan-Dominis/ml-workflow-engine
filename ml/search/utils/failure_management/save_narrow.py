import json
import logging
from pathlib import Path

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)


def save_narrow(
    *,
    narrow_result: dict,
    best_params: dict,
    tgt_file: Path
) -> None:
    narrow_info = {
        "narrow_result": narrow_result,
        "best_params": best_params
    }

    try:
        with tgt_file.open("w", encoding="utf-8") as f:
            json.dump(narrow_info, f, indent=2)
        logger.info(f"Narrow search done marker written to {tgt_file}.")
    except Exception as e:
        msg = f"Failed to save narrow search done marker to {tgt_file}"
        logger.exception(msg)
        raise PersistenceError(msg) from e