"""Operator hashing and validation utilities for feature freezing integrity."""

import hashlib
import inspect
import json
import logging
import sys

import pandas as pd
import sklearn
from ml.exceptions import DataError, UserError
from ml.registries.catalogs import FEATURE_OPERATORS

logger = logging.getLogger(__name__)

def generate_operator_hash(operator_names):
    """Generate deterministic hash for operator set and runtime context.

    Args:
        operator_names: Operator names included in the freeze process.

    Returns:
        str: Deterministic operator hash.
    """

    operator_names = sorted(operator_names)
    operators = [FEATURE_OPERATORS[name] for name in operator_names]
    operator_sources = ""
    for operator in operators:
        source = inspect.getsource(operator)
        operator_sources += source

    operator_ids = [f"{op.__module__}.{op.__name__}" for op in operators]

    payload = {
        "operators": operator_sources,
        "operator_ids": operator_ids,
        "python": sys.version,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }

    operator_hash = hashlib.md5(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()

    return operator_hash

def validate_operators(operators: list, operator_hash: str):
    """Validate operator names and verify computed hash matches expected hash.

    Args:
        operators: Operator names to validate.
        operator_hash: Expected operator hash value.

    Returns:
        None: Raises on validation/hash mismatch failure.
    """

    for name in operators:
        if name not in FEATURE_OPERATORS:
            raise UserError(f"Unknown operator: {name}")

    generated_hash = generate_operator_hash(operators)
    if generated_hash != operator_hash:
        msg = f"Operator hash mismatch: expected {operator_hash}, got {generated_hash}"
        logger.error(msg)
        raise DataError(msg)
    logger.debug(f"Operators validated successfully with hash: {generated_hash}")
