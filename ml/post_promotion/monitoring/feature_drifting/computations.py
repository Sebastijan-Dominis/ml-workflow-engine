"""Computations for feature drifting monitoring."""
import logging
from typing import Any

import numpy as np
import pandas as pd

from ml.exceptions import MonitoringError, RuntimeMLError
from ml.post_promotion.monitoring.feature_drifting.analysis import (
    analyze_ks_result,
    analyze_psi_result,
)
from ml.post_promotion.monitoring.feature_drifting.utils import infer_drift_method

logger = logging.getLogger(__name__)

def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index (PSI) for feature drifting.

    Args:
        expected: A pandas Series representing the expected distribution.
        actual: A pandas Series representing the actual distribution.
        bins: Number of bins to use for numeric features.

    Returns:
        A float representing the PSI value."""
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        msg = "Expected and actual distributions must have at least one non-null value."
        logger.error(msg)
        raise MonitoringError(msg)

    if pd.api.types.is_numeric_dtype(expected):
        bin_edges = np.histogram_bin_edges(expected, bins=bins)
        bin_edges = bin_edges.tolist()  # Convert numpy array to list for pd.cut

        expected_bins = pd.cut(expected, bins=bin_edges, include_lowest=True)
        actual_bins = pd.cut(actual, bins=bin_edges, include_lowest=True)
    else:
        expected_bins = expected.astype(str)
        actual_bins = actual.astype(str)

    e_dist = expected_bins.value_counts(normalize=True)
    a_dist = actual_bins.value_counts(normalize=True)

    all_bins = set(e_dist.index).union(a_dist.index)

    psi_val = 0.0
    for b in all_bins:
        e = e_dist.get(b, 1e-6)
        a = a_dist.get(b, 1e-6)

        # prevent log(0)
        if e == 0:
            e = 1e-6
        if a == 0:
            a = 1e-6

        psi_val += (e - a) * np.log(e / a)

    return float(psi_val)

def compute_ks(expected: pd.Series, actual: pd.Series) -> float:
    """Kolmogorov-Smirnov statistic for numeric features.

    Args:
        expected: A pandas Series representing the expected distribution.
        actual: A pandas Series representing the actual distribution.

    Returns:
        A float representing the KS statistic.
    """
    from scipy.stats import ks_2samp
    result: Any = ks_2samp(expected, actual)
    stat = getattr(result, "statistic", None)
    if stat is None:
        # scipy may return a tuple (statistic, pvalue) in some versions
        stat = result[0]
    return float(stat)

def compute_drift(
    expected: pd.Series,
    actual: pd.Series
) -> float:
    """Compute drift between expected and actual distributions.

    Args:
        expected: A pandas Series representing the expected distribution.
        actual: A pandas Series representing the actual distribution.

    Returns:
        A float representing the computed drift metric (KS statistic or PSI value).
    """
    method = infer_drift_method(expected)

    if method == "ks":
        try:
            result = compute_ks(expected, actual)
        except Exception as e:
            msg = f"Error computing KS statistic for feature: {expected.name}"
            logger.exception(msg)
            raise MonitoringError(msg) from e
        feature_name = str(expected.name) if expected.name is not None else "<unknown>"
        analyze_ks_result(feature_name, result)
    elif method == "psi":
        try:
            result = compute_psi(expected, actual)
        except Exception as e:
            msg = f"Error computing PSI for feature: {expected.name}"
            logger.exception(msg)
            raise MonitoringError(msg) from e
        feature_name = str(expected.name) if expected.name is not None else "<unknown>"
        analyze_psi_result(feature_name, result)
    else:
        msg = f"Unsupported drift method: {method}"
        logger.error(msg)
        raise RuntimeMLError(msg)
    return result
