import numpy as np
import pandas as pd
from ml.post_promotion.inference.hashing.input_row import hash_input_row


def test_hash_input_row_handles_nan_and_floats() -> None:
    s1 = pd.Series([np.nan, 0.1, "a"], index=["x", "y", "z"])
    s2 = pd.Series([np.nan, 0.1, "a"], index=["x", "y", "z"])
    h1 = hash_input_row(s1)
    h2 = hash_input_row(s2)
    assert h1 == h2

    s3 = pd.Series([np.nan, 0.10000000000000001, "a"], index=["x", "y", "z"])
    assert hash_input_row(s3) == h1
