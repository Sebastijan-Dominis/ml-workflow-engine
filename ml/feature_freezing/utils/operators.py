"""Operator hashing utility for operator integrity."""

import hashlib
import inspect
import json
import logging
import sys

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

    bytecode_parts = []
    operator_ids = []

    for op in operators:
        operator_ids.append(f"{op.__module__}.{op.__name__}")

        # hash all functions defined on the class
        for _name, member in inspect.getmembers(op, inspect.isfunction):
            bytecode_parts.append(member.__code__.co_code)

    payload = {
        "operator_ids": operator_ids,
        "operator_bytecode": b"".join(bytecode_parts).hex(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    }

    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

