"""Unit tests for target strategies and shared validation behavior."""

from __future__ import annotations

import pandas as pd
import pytest
from ml.exceptions import UserError
from ml.targets.adr.v1 import AdrTargetV1
from ml.targets.base import TargetStrategy
from ml.targets.cancellation.v1 import CancellationTargetV1
from ml.targets.lead_time.v1 import LeadTimeTargetV1
from ml.targets.no_show.v1 import NoShowTargetV1
from ml.targets.repeated_guest.v1 import RepeatedGuestTargetV1
from ml.targets.room_upgrade.v1 import RoomUpgradeTargetV1
from ml.targets.special_requests.v1 import SpecialRequestsTargetV1

pytestmark = pytest.mark.unit


class _EchoStrategy(TargetStrategy):
    """Minimal concrete strategy used to exercise base-class validation paths."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a simple projection so tests can assert build delegation."""
        return data[["row_id"]].copy()


@pytest.mark.parametrize(
    ("strategy", "target_col"),
    [
        pytest.param(AdrTargetV1, "adr", id="adr"),
        pytest.param(CancellationTargetV1, "is_canceled", id="cancellation"),
        pytest.param(LeadTimeTargetV1, "lead_time", id="lead-time"),
        pytest.param(RepeatedGuestTargetV1, "is_repeated_guest", id="repeated-guest"),
        pytest.param(SpecialRequestsTargetV1, "total_of_special_requests", id="special-requests"),
    ],
)
def test_passthrough_target_strategies_return_expected_columns_and_copy(
    strategy: type[TargetStrategy],
    target_col: str,
) -> None:
    """Project the configured target column and isolate output from later input mutations."""
    source = pd.DataFrame(
        {
            "row_id": ["r1", "r2"],
            target_col: [10, 20],
            "unused": [0, 1],
        }
    )

    result = strategy().build(source)

    assert result.columns.tolist() == [target_col, "row_id"]
    assert result[target_col].tolist() == [10, 20]
    source.loc[0, target_col] = 999
    assert result[target_col].tolist() == [10, 20]


def test_target_strategy_base_validation_rejects_frames_without_row_id() -> None:
    """Raise ``UserError`` before strategy-specific logic when ``row_id`` is missing."""
    source = pd.DataFrame({"some_feature": [1, 2, 3]})

    with pytest.raises(UserError, match="Target data missing required columns"):
        _EchoStrategy().build(source)


def test_no_show_target_builds_binary_flags_from_reservation_status() -> None:
    """Map ``reservation_status`` to deterministic binary ``no_show`` labels."""
    source = pd.DataFrame(
        {
            "row_id": ["a", "b", "c"],
            "reservation_status": ["No-Show", "Check-Out", "Canceled"],
        }
    )

    result = NoShowTargetV1().build(source)

    assert result.columns.tolist() == ["no_show", "row_id"]
    assert result["no_show"].tolist() == [1, 0, 0]
    assert result["row_id"].tolist() == ["a", "b", "c"]


def test_room_upgrade_target_compares_room_types_after_string_coercion() -> None:
    """Treat different raw dtypes equivalently by comparing string-normalized room values."""
    source = pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3", "r4"],
            "reserved_room_type": ["A", 1, None, "B"],
            "assigned_room_type": ["A", "1", "None", "C"],
        }
    )

    result = RoomUpgradeTargetV1().build(source)

    assert result.columns.tolist() == ["room_upgrade", "row_id"]
    assert result["room_upgrade"].tolist() == [0, 0, 0, 1]


def test_target_strategies_surface_missing_strategy_specific_columns_as_key_error() -> None:
    """Bubble up pandas ``KeyError`` when strategy-specific target columns are absent."""
    source = pd.DataFrame({"row_id": ["r1", "r2"]})

    with pytest.raises(KeyError, match="is_canceled"):
        CancellationTargetV1().build(source)
