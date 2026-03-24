"""Unit tests for the DAG builder used to order dataset merges.

These tests exercise the logic in `ml.data.merge.merge_dataset_into_main.build_dataset_dag`.
They validate that datasets which share merge keys are ordered so that earlier
datasets (by list position) precede later ones when a key overlap exists, and
that non-overlapping datasets are both present in the result.
"""

from types import SimpleNamespace

import pytest

from typing import cast
from ml.feature_freezing.freeze_strategies.tabular.config.models import DatasetConfig
from ml.data.merge.merge_dataset_into_main import build_dataset_dag


pytestmark = pytest.mark.unit


def _ds(name: str, merge_key):
    """Helper to build lightweight dataset-like objects for DAG construction.

    The real function only needs `name` and `merge_key` attributes; using
    SimpleNamespace keeps the tests focused and avoids pydantic overhead.
    """
    return SimpleNamespace(name=name, merge_key=merge_key)


def test_build_dataset_dag_orders_overlapping_keys() -> None:
    """Datasets that share a merge key should produce an order where the
    earlier-listed dataset precedes the later one.
    """
    a = _ds("a", "id")
    b = _ds("b", "id")

    order = build_dataset_dag(cast(list[DatasetConfig], [a, b]))

    assert order == ["a", "b"]


def test_build_dataset_dag_handles_list_merge_keys_and_partial_overlap() -> None:
    """When merge_key is a list, intersections should still be detected and
    respected in the ordering.
    """
    # a shares key 'x' with b; b also has key 'y' not present in a.
    a = _ds("a", ["x"])
    b = _ds("b", ["x", "y"])  # overlaps with a on 'x'
    c = _ds("c", "z")

    order = build_dataset_dag(cast(list[DatasetConfig], [a, b, c]))

    # a must come before b; c is independent but should be included
    assert "a" in order and "b" in order and "c" in order
    assert order.index("a") < order.index("b")


def test_build_dataset_dag_returns_all_nodes_when_no_overlap() -> None:
    """If no datasets share keys, the DAG contains all nodes and no edges.

    We check that the returned list contains every dataset name exactly once.
    """
    a = _ds("a", "id")
    b = _ds("b", "other")

    order = build_dataset_dag(cast(list[DatasetConfig], [a, b]))

    assert set(order) == {"a", "b"}
    assert len(order) == 2
