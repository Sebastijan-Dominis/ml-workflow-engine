from dataclasses import dataclass


@dataclass
class DataStats:
    n_samples: int
    class_counts: dict[int, int]
    minority_ratio: float
