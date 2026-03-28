import pandas as pd
from ml.post_promotion.inference.hashing.input_row import hash_input_row


def test_hash_input_row_nan_and_order_independence():
    s1 = pd.Series({"a": 1.23456789, "b": float("nan"), "c": "x"})
    s2 = pd.Series({"c": "x", "b": float("nan"), "a": 1.23456789})

    h1 = hash_input_row(s1)
    h2 = hash_input_row(s2)

    assert isinstance(h1, str) and len(h1) == 64
    assert h1 == h2


def test_hash_input_row_handles_floats_and_nulls():
    s = pd.Series({"a": 1.0, "b": 1.0000000001, "c": None})
    h = hash_input_row(s)
    assert isinstance(h, str) and len(h) == 64
