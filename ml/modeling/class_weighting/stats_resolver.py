"""Utilities for computing class-distribution statistics from targets."""

import logging
from collections import Counter

from ml.modeling.class_weighting.models import DataStats

logger = logging.getLogger(__name__)

def compute_data_stats(y):
    """Compute sample count, class counts, and minority ratio for target labels.

    Args:
        y: Target label sequence.

    Returns:
        Data statistics object with sample size, class counts, and minority ratio.
    """

    counts = Counter(y)
    n = len(y)
    minority_ratio = min(counts.values()) / n if n > 0 else 0
    logger.info("Computed data stats: n_samples=%d class_counts=%s minority_ratio=%.4f",
        n, dict(counts), minority_ratio)
    return DataStats(
        n_samples=n,
        class_counts=dict(counts),
        minority_ratio=minority_ratio
    )
