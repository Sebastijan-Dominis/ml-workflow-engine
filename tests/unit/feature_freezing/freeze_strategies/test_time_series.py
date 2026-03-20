"""Unit tests for placeholder time-series freeze strategy behavior."""

import pytest
from ml.feature_freezing.freeze_strategies.time_series import FreezeTimeSeries

pytestmark = pytest.mark.unit


def test_freeze_time_series_raises_not_implemented_error() -> None:
    """Raise explicit NotImplementedError until strategy implementation exists."""
    strategy = FreezeTimeSeries()

    with pytest.raises(NotImplementedError, match="not implemented"):
        strategy.freeze(
            config={},
            snapshot_id="snapshot-1",
            timestamp="2026-03-05T00:00:00",
            start_time=0.0,
            owner="tests",
            snapshot_binding_key=None,
        )
