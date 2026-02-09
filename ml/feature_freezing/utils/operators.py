import hashlib
import inspect
import json
import logging
import sys

import pandas as pd
import sklearn

from ml.exceptions import DataError, UserError
from ml.registry.feature_operators import FEATURE_OPERATORS

logger = logging.getLogger(__name__)

def generate_operator_hash(operator_names):
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

    operators_hash = hashlib.md5(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()

    return operators_hash

def validate_operators(operators: list, operator_hash: str):
    for name in operators:
        if name not in FEATURE_OPERATORS:
            raise UserError(f"Unknown operator: {name}")
        
    generated_hash = generate_operator_hash(operators)
    if generated_hash != operator_hash:
        msg = f"Operator hash mismatch: expected {operator_hash}, got {generated_hash}"
        logger.error(msg)
        raise DataError(msg)