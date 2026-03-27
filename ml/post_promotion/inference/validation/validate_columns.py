import pandas as pd

from ml.post_promotion.inference.classes.predictions_schema import (
    BASE_EXPECTED_COLUMNS,
    PROBA_PREFIX,
)


def validate_columns(df: pd.DataFrame) -> list[str]:
    cols = set(df.columns)

    # Check required base columns
    missing = set(BASE_EXPECTED_COLUMNS) - cols
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check probability columns (optional but must follow pattern)
    proba_cols = [c for c in cols if c.startswith(PROBA_PREFIX)]

    if proba_cols:
        # Ensure consistent indexing (proba_0, proba_1, ...)
        expected = {f"{PROBA_PREFIX}{i}" for i in range(len(proba_cols))}
        if set(proba_cols) != expected:
            raise ValueError(f"Probability columns malformed: {proba_cols}")

    return list(cols)
