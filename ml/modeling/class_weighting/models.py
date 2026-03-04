"""Data models used by class-weighting and scoring policy resolution."""

from dataclasses import dataclass


@dataclass
class DataStats:
    """Summary statistics for target distribution and class imbalance."""

    n_samples: int
    class_counts: dict[int, int]
    minority_ratio: float
