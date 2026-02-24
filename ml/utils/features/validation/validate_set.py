import logging

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def validate_set(hash_type: str, hashes: set, feature_sets: list) -> None:
    if len(hashes) != 1:
        msg = f"{hash_type} hashes do not match across feature sets. Feature sets involved: " + ", ".join(
            [f"{feature_sets[i].name} (version: {feature_sets[i].version})" for i in range(len(feature_sets))]
        )
        logger.error(msg)
        raise DataError(msg)