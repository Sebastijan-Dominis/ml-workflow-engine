import pandas as pd
import pytest
import scipy.stats as stats
from ml.exceptions import MonitoringError, RuntimeMLError
from ml.post_promotion.monitoring.feature_drifting import computations as comp_mod


def test_compute_psi_empty_raises():
    with pytest.raises(MonitoringError):
        comp_mod.compute_psi(pd.Series([]), pd.Series([1, 2, 3]))


def test_compute_psi_numeric_returns_float():
    expected = pd.Series([1, 2, 2, 3, 4, 5])
    actual = pd.Series([1, 1, 2, 3, 4, 6])
    val = comp_mod.compute_psi(expected, actual, bins=4)
    assert isinstance(val, float)
    assert val >= 0.0


def test_compute_ks_identical_arrays():
    s = pd.Series([1, 2, 3, 4, 5])
    stat = comp_mod.compute_ks(s, s)
    assert pytest.approx(stat, rel=1e-6) == 0.0


def test_compute_drift_uses_ks_for_numeric(monkeypatch):
    # force analyze_ks_result to be a no-op to avoid threshold checks
    monkeypatch.setattr(comp_mod, "analyze_ks_result", lambda name, v: None)

    expected = pd.Series(range(30), name="feat")
    actual = pd.Series(range(30), name="feat")

    res = comp_mod.compute_drift(expected, actual)
    assert isinstance(res, float)


def test_compute_psi_non_numeric_returns_float():
    expected = pd.Series(["a", "a", "b", "c"])
    actual = pd.Series(["a", "b", "b", "c"])
    val = comp_mod.compute_psi(expected, actual)
    assert isinstance(val, float)


def test_compute_drift_uses_psi_for_low_cardinality(monkeypatch):
    monkeypatch.setattr(comp_mod, "analyze_psi_result", lambda name, v: None)

    expected = pd.Series([1, 1, 2, 2, 3], name="feat")
    actual = pd.Series([1, 2, 2, 3, 3], name="feat")

    res = comp_mod.compute_drift(expected, actual)
    assert isinstance(res, float)


def test_compute_drift_ks_exception_handling(monkeypatch):
    def fake_compute_ks(e, a):
        raise ValueError("boom")

    monkeypatch.setattr(comp_mod, "compute_ks", fake_compute_ks)

    expected = pd.Series(range(30))
    actual = pd.Series(range(30))

    with pytest.raises(MonitoringError):
        comp_mod.compute_drift(expected, actual)


def test_compute_drift_unsupported_method(monkeypatch):
    monkeypatch.setattr(comp_mod, "infer_drift_method", lambda s: "unsupported")

    expected = pd.Series([1, 2, 3])
    actual = pd.Series([1, 2, 3])

    with pytest.raises(RuntimeMLError):
        comp_mod.compute_drift(expected, actual)


def test_compute_drift_unknown_feature_name_ks(monkeypatch):
    # ensure feature_name falls back to <unknown>
    monkeypatch.setattr(comp_mod, "analyze_ks_result", lambda name, v: None)

    expected = pd.Series(range(30))  # no name
    actual = pd.Series(range(30))

    res = comp_mod.compute_drift(expected, actual)
    assert isinstance(res, float)


def test_compute_ks_handles_tuple_result(monkeypatch):
    # simulate scipy.stats.ks_2samp returning a tuple
    monkeypatch.setattr(stats, "ks_2samp", lambda e, a: (0.1234, 0.5))
    s = pd.Series([1, 2, 3])
    stat = comp_mod.compute_ks(s, s)
    assert pytest.approx(stat, rel=1e-6) == 0.1234
