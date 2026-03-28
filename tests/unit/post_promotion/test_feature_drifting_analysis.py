import pytest
from ml.exceptions import RuntimeMLError
from ml.post_promotion.monitoring.feature_drifting.analysis import (
    analyze_ks_result,
    analyze_psi_result,
)


def test_analyze_ks_result_out_of_bounds_raises():
    with pytest.raises(RuntimeMLError):
        analyze_ks_result("feat", -0.01)
    with pytest.raises(RuntimeMLError):
        analyze_ks_result("feat", 1.5)


def test_analyze_ks_result_valid_ranges_do_not_raise():
    for v in [0.05, 0.2, 0.4, 0.8]:
        analyze_ks_result("feat", v)


def test_analyze_psi_result_invalid_inputs_raise():
    with pytest.raises(RuntimeMLError):
        analyze_psi_result("feat", -0.1)
    with pytest.raises(RuntimeMLError):
        analyze_psi_result("feat", float("inf"))


def test_analyze_psi_result_valid_ranges_do_not_raise():
    for v in [0.05, 0.2, 0.4, 0.8]:
        analyze_psi_result("feat", v)
