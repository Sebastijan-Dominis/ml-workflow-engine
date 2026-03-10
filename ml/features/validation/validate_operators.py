import logging

from ml.exceptions import DataError, UserError
from ml.feature_freezing.utils.operators import generate_operator_hash
from ml.registries.catalogs import FEATURE_OPERATORS

logger = logging.getLogger(__name__)

def validate_operators(operators: list, operator_hash: str | None, file_path: str | None = None) -> None:
    """Validate operator names and verify computed hash matches expected hash.

    Args:
        operators: Operator names to validate.
        operator_hash: Expected operator hash value.
        file_path: Optional file path for logging context in case of validation failure.

    Returns:
        None: Raises on validation/hash mismatch failure.
    """

    if file_path:
        logger.debug(f"Validating operators for feature set at {file_path} with expected operator hash {operator_hash} and operators {operators}")
    else:
        logger.debug(f"Validating operators with expected operator hash {operator_hash} and operators {operators}; no file path provided for context.")

    if operator_hash is None or operator_hash == "none":
        if operators:
            msg = f"Operator hash is required when operators are specified. Provided operators: {operators}"
            logger.error(msg)
            raise DataError(msg)
        else:
            logger.debug("No operators specified and no operator hash provided; skipping operator validation.")
            return

    for name in operators:
        if name not in FEATURE_OPERATORS:
            raise UserError(f"Unknown operator: {name}")

    generated_hash = generate_operator_hash(operators)
    if generated_hash != operator_hash:
        msg = f"Operator hash mismatch: expected {operator_hash}, got {generated_hash}"
        logger.error(msg)
        raise DataError(msg)
    logger.debug(f"Operators validated successfully with hash: {generated_hash}")
