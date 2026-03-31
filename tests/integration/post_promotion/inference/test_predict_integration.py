import pandas as pd
import pytest
from ml.exceptions import InferenceError
from ml.post_promotion.inference.execution.predict import predict


def test_predict_raises_inference_error_on_failure() -> None:
    class BadArtifact:
        def predict(self, X):
            raise RuntimeError("boom")

    X = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(InferenceError):
        predict(X, BadArtifact())


def test_predict_returns_empty_proba_when_no_predict_proba() -> None:
    class ArtifactNoProba:
        def predict(self, X):
            return [0, 1]

    X = pd.DataFrame({"a": [1, 2]})
    preds, proba = predict(X, ArtifactNoProba())
    assert preds.tolist() == [0, 1]
    assert proba.empty
