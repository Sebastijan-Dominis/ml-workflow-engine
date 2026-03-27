"""Analysis functions for feature drifting monitoring."""
import logging

import numpy as np

from ml.exceptions import RuntimeMLError

logger = logging.getLogger(__name__)

def analyze_ks_result(feature_name: str, ks_stat: float) -> None:
    """Analyze KS statistic result and log appropriate messages.

    Args:
        feature_name: Name of the feature being analyzed.
        ks_stat: The KS statistic value to analyze.
    """
    if ks_stat < 0 or ks_stat > 1:
        msg = f"KS statistic out of bounds [0,1]: {ks_stat}"
        logger.error(msg)
        raise RuntimeMLError(msg)
    if ks_stat < 0.1:
        logger.debug(f"KS statistic {ks_stat:.4f} indicates low drift for feature: {feature_name}")
    elif ks_stat < 0.25:
        logger.warning(f"KS statistic {ks_stat:.4f} indicates moderate drift for feature: {feature_name}")
    elif ks_stat < 0.5:
        logger.error(f"KS statistic {ks_stat:.4f} indicates high drift for feature: {feature_name}")
    else:
        logger.critical(f"KS statistic {ks_stat:.4f} indicates severe drift for feature: {feature_name}")

def analyze_psi_result(feature_name: str, psi_val: float) -> None:
    """Analyze PSI value and log appropriate messages.

    Args:
        feature_name: Name of the feature being analyzed.
        psi_val: The PSI value to analyze.
    """
    if psi_val < 0 or not np.isfinite(psi_val):
        msg = f"PSI value is invalid (negative or non-finite): {psi_val}"
        logger.error(msg)
        raise RuntimeMLError(msg)
    if psi_val < 0.1:
        logger.debug(f"PSI value {psi_val:.4f} indicates low drift for feature: {feature_name}")
    elif psi_val < 0.25:
        logger.warning(f"PSI value {psi_val:.4f} indicates moderate drift for feature: {feature_name}")
    elif psi_val < 0.5:
        logger.error(f"PSI value {psi_val:.4f} indicates high drift for feature: {feature_name}")
    else:
        logger.critical(f"PSI value {psi_val:.4f} indicates severe drift for feature: {feature_name}")
