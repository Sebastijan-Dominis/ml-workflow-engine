"""Unit tests for validated dataset merging into the main frame."""

from pathlib import Path

import pandas as pd
import pytest
from ml.data.merge.merge_dataset_into_main import merge_dataset_into_main
from ml.exceptions import DataError

pytestmark = pytest.mark.unit


def test_merge_dataset_into_main_raises_when_merge_key_missing_in_dataset() -> None:
    """Reject dataset merges when incoming dataset does not have required merge key column."""
    main_df = pd.DataFrame({"row_id": [1, 2]})
    incoming_df = pd.DataFrame({"other": [10, 20]})

    with pytest.raises(DataError, match="missing merge key 'row_id'"):
        merge_dataset_into_main(
            main_df,
            incoming_df,
            merge_key="row_id",
            dataset_name="booking_context_features",
            dataset_version="v1",
            dataset_snapshot_path=Path("snapshot"),
            dataset_path=Path("dataset.parquet"),
        )


def test_merge_dataset_into_main_raises_when_main_lacks_merge_key_and_not_empty() -> None:
    """Reject non-initial merges when main dataframe lacks key required for alignment."""
    main_df = pd.DataFrame({"not_row_id": [1]})
    incoming_df = pd.DataFrame({"row_id": [1], "feature": [3.2]})

    with pytest.raises(DataError, match="not found in the main dataset"):
        merge_dataset_into_main(
            main_df,
            incoming_df,
            merge_key="row_id",
            dataset_name="pricing_party_features",
            dataset_version="v2",
            dataset_snapshot_path=Path("snapshot"),
            dataset_path=Path("dataset.parquet"),
        )


def test_merge_dataset_into_main_returns_incoming_data_when_main_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use incoming dataframe as main dataset for first merge while still validating metadata/hash."""
    main_df = pd.DataFrame()
    incoming_df = pd.DataFrame({"row_id": [1, 2], "feature": [0.5, 1.0]})

    monkeypatch.setattr(
        "ml.data.merge.merge_dataset_into_main.load_json",
        lambda path: {"data": {"path_suffix": "x", "format": "parquet"}},
    )
    monkeypatch.setattr(
        "ml.data.merge.merge_dataset_into_main.validate_data",
        lambda **kwargs: "validated-hash-123",
    )

    merged, data_hash = merge_dataset_into_main(
        main_df,
        incoming_df,
        merge_key="row_id",
        dataset_name="customer_history_features",
        dataset_version="v1",
        dataset_snapshot_path=Path("snapshot-001"),
        dataset_path=Path("dataset.parquet"),
    )

    assert merged.equals(incoming_df)
    assert data_hash == "validated-hash-123"


def test_merge_dataset_into_main_drops_overlapping_non_key_columns_before_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop non-key overlaps in incoming dataset to avoid duplicate column conflicts on merge."""
    main_df = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "hotel": ["A", "B", "C"],
            "main_only": [10, 20, 30],
        }
    )
    incoming_df = pd.DataFrame(
        {
            "row_id": [2, 3, 4],
            "hotel": ["B_new", "C_new", "D_new"],
            "incoming_only": [200, 300, 400],
        }
    )

    monkeypatch.setattr("ml.data.merge.merge_dataset_into_main.load_json", lambda path: {})
    monkeypatch.setattr("ml.data.merge.merge_dataset_into_main.validate_data", lambda **kwargs: "hash-xyz")

    merged, data_hash = merge_dataset_into_main(
        main_df,
        incoming_df,
        merge_key="row_id",
        dataset_name="room_allocation_features",
        dataset_version="v3",
        dataset_snapshot_path=Path("snapshot-003"),
        dataset_path=Path("dataset.parquet"),
    )

    assert data_hash == "hash-xyz"
    assert merged["row_id"].tolist() == [2, 3]
    assert merged["hotel"].tolist() == ["B", "C"]
    assert merged["incoming_only"].tolist() == [200, 300]


def test_merge_dataset_into_main_raises_when_inner_merge_result_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise DataError when key alignment produces an empty merged dataframe."""
    main_df = pd.DataFrame({"row_id": [1, 2], "x": [10, 20]})
    incoming_df = pd.DataFrame({"row_id": [9, 10], "y": [90, 100]})

    monkeypatch.setattr("ml.data.merge.merge_dataset_into_main.load_json", lambda path: {})
    monkeypatch.setattr("ml.data.merge.merge_dataset_into_main.validate_data", lambda **kwargs: "hash-xyz")

    with pytest.raises(DataError, match="Merged dataset is empty"):
        merge_dataset_into_main(
            main_df,
            incoming_df,
            merge_key="row_id",
            dataset_name="channel_features",
            dataset_version="v5",
            dataset_snapshot_path=Path("snapshot-005"),
            dataset_path=Path("dataset.parquet"),
        )
