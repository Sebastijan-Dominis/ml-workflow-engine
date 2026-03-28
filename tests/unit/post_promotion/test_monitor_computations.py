import numpy as np
import pandas as pd
from ml.post_promotion.monitoring.feature_drifting.computations import compute_ks, compute_psi
from ml.post_promotion.monitoring.feature_drifting.utils import infer_drift_method


def test_infer_drift_method_for_numeric_and_small_cardinality():
    ser = pd.Series([1.0, 2.0, 3.0])
    # Low cardinality numeric series should be treated as categorical → PSI
    assert infer_drift_method(ser) == "psi"


def test_infer_drift_method_for_high_cardinality_object():
    ser = pd.Series([f"s{i}" for i in range(200)])
    assert infer_drift_method(ser) == "psi"


def test_compute_ks_basic():
    a = pd.Series(np.random.normal(size=100))
    b = pd.Series(a.to_numpy() + 0.1)
    res = compute_ks(a, b)
    # KS wrapper returns the statistic (float)
    assert isinstance(res, float)


def test_compute_psi_basic():
    a = pd.Series(np.random.choice(["x", "y", "z"], size=200))
    b = pd.Series(np.random.choice(["x", "y", "z"], size=200))
    res = compute_psi(a, b)
    assert isinstance(res, float)
