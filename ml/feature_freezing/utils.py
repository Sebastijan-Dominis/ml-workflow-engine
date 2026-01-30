import hashlib
import inspect
import sys
import pandas as pd
import sklearn
import json

from ml.registry import FEATURE_OPERATORS

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