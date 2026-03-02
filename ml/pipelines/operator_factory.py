"""Factory helpers for creating feature engineering operator instances."""

import pandas as pd

from ml.registry.feature_operators import FEATURE_OPERATORS


def build_operators(derived_schema: pd.DataFrame) -> dict[str, object]:
    """Instantiate operators declared in ``derived_schema``.

    Args:
        derived_schema: DataFrame containing a ``source_operator`` column.

    Returns:
        dict[str, object]: Mapping of operator name to instantiated operator.
    """
    return {name: FEATURE_OPERATORS[name]() for name in derived_schema["source_operator"].unique()}
