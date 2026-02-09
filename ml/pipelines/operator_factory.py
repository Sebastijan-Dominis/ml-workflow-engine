from typing import Dict

import pandas as pd

from ml.registry.feature_operators import FEATURE_OPERATORS


def build_operators(derived_schema: pd.DataFrame) -> Dict[str, object]:
    """
    Instantiate all feature operators based on derived_schema['source_operator'].
    Returns a dict {operator_name: operator_instance}
    """
    return {name: FEATURE_OPERATORS[name]() for name in derived_schema["source_operator"].unique()}
