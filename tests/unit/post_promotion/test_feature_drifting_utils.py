import pandas as pd
from ml.post_promotion.monitoring.feature_drifting.utils import infer_drift_method


def test_infer_drift_method_numeric_high_cardinality():
    s = pd.Series(range(100))
    assert infer_drift_method(s) == "ks"


def test_infer_drift_method_numeric_low_cardinality():
    s = pd.Series([1, 1, 2, 2])
    assert infer_drift_method(s) == "psi"


def test_infer_drift_method_datetime_and_object():
    dt = pd.Series(pd.date_range("2020-01-01", periods=3))
    assert infer_drift_method(dt) == "psi"

    obj = pd.Series(["a", "b", "a"])
    assert infer_drift_method(obj) == "psi"
