from collections import Counter
import logging

from ml.utils.experiments.class_weights.models import DataStats

logger = logging.getLogger(__name__)

def compute_data_stats(y):
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