import numpy as np
import pandas as pd
import pytest
from ml.exceptions import InferenceError
from ml.post_promotion.inference.execution.predict import predict
from ml.post_promotion.inference.hashing.input_row import hash_input_row


class DummyModel:
    def __init__(self, preds, probas=None, raise_on_predict: bool = False):
        self._preds = preds
        self._probas = probas
        self._raise = raise_on_predict

    def predict(self, X: pd.DataFrame):
        if self._raise:
            raise ValueError("predict failed")
        return list(self._preds)

    def predict_proba(self, X: pd.DataFrame):
        if self._probas is None:
            raise AttributeError("no proba")
        return self._probas


def test_predict_with_and_without_proba():
    X = pd.DataFrame({"f": [1, 2]})

    model_with_proba = DummyModel(preds=[0, 1], probas=[[0.8, 0.2], [0.3, 0.7]])
    preds, proba = predict(X, model_with_proba)
    assert list(preds) == [0, 1]
    assert not proba.empty

    class NoProba:
        def predict(self, X):
            return [1, 0]

    preds2, proba2 = predict(X, NoProba())
    assert list(preds2) == [1, 0]
    assert proba2.empty


def test_predict_raises_wrapped_inference_error():
    X = pd.DataFrame({"f": [1]})
    broken = DummyModel(preds=None, raise_on_predict=True)
    with pytest.raises(InferenceError):
        predict(X, broken)


def test_hash_input_row_stability_and_nan_and_float_format():
    # Order should not matter
    a = pd.Series({"b": np.nan, "a": 1})
    b = pd.Series({"a": 1, "b": np.nan})
    assert hash_input_row(a) == hash_input_row(b)

    # Float formatting should be stable
    s1 = pd.Series({"x": 0.10000000000000001})
    s2 = pd.Series({"x": 0.1})
    assert hash_input_row(s1) == hash_input_row(s2)

    h = hash_input_row(pd.Series({"v": 3.14159}))
    assert isinstance(h, str) and len(h) == 64
