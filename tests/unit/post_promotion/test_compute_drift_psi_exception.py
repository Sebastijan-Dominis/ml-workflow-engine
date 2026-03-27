import pandas as pd
import pytest
from ml.exceptions import MonitoringError
from ml.post_promotion.monitoring.feature_drifting.computations import compute_drift


def test_compute_drift_psi_exception_raised_when_expected_is_all_nan():
    expected = pd.Series([float("nan"), float("nan")], name="my_feature")
    actual = pd.Series([1.0, 2.0], name="my_feature")

    with pytest.raises(MonitoringError) as excinfo:
        compute_drift(expected, actual)

    assert "Error computing PSI for feature: my_feature" in str(excinfo.value)
    # ensure the original cause indicates empty distributions
    cause = excinfo.value.__cause__
    assert cause is not None
    assert "Expected and actual distributions must have at least one non-null value." in str(cause)
