import logging
from pathlib import Path
from typing import Any

from ml.exceptions import UserError
from ml.registry.artifact_hashing import HASH_ARTIFACT_REGISTRY

logger = logging.getLogger(__name__)

def hash_artifact(obj: Any, method="sha256", temp_dir: Path | None = None) -> str:
    """
    Hash a model or pipeline artifact. Dispatches to the correct serializer via registry.
    Raises UserError for unknown types.
    """
    for t, fn in HASH_ARTIFACT_REGISTRY.items():
        if isinstance(obj, t):
            return fn(obj, method=method, temp_dir=temp_dir)

    msg = f"No hashing function registered for type {type(obj)}. Consider adding to HASH_ARTIFACT_REGISTRY."
    logger.error(msg)
    raise UserError(msg)