"""Unit tests for target-distribution statistics resolution."""

from ml.modeling.class_weighting.stats_resolver import compute_data_stats


def test_compute_data_stats_returns_expected_counts_and_minority_ratio() -> None:
    """Compute sample stats from binary labels and verify class-balance metadata."""
    stats = compute_data_stats([0, 1, 1, 0, 1])

    assert stats.n_samples == 5
    assert stats.class_counts == {0: 2, 1: 3}
    assert stats.minority_ratio == 0.4


def test_compute_data_stats_handles_empty_target_sequence() -> None:
    """Return empty-count stats with a zero minority ratio for empty targets."""
    stats = compute_data_stats([])

    assert stats.n_samples == 0
    assert stats.class_counts == {}
    assert stats.minority_ratio == 0
