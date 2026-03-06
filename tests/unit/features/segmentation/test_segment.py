"""Unit tests for dataset segmentation helpers."""

import importlib
import sys
import types
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from ml.config.schemas.model_specs import SegmentationConfig, SegmentationFilter
from ml.exceptions import DataError, UserError

pytestmark = pytest.mark.unit


def _import_segment_with_stubbed_catalogs() -> types.ModuleType:
    """Import segmentation module with isolated operator-catalog dependency."""
    module_name = "ml.features.segmentation.segment"
    registries_name = "ml.registries"
    catalogs_name = "ml.registries.catalogs"

    sys.modules.pop(module_name, None)

    fake_registries = types.ModuleType(registries_name)
    fake_registries.__path__ = []
    sys.modules[registries_name] = fake_registries

    fake_catalogs = types.ModuleType(catalogs_name)
    fake_catalogs.__dict__["OP_MAP"] = {
        "eq": lambda s, v: s == v,
        "neq": lambda s, v: s != v,
        "in": lambda s, v: s.isin(v),
        "not_in": lambda s, v: ~s.isin(v),
        "gt": lambda s, v: s > v,
        "gte": lambda s, v: s >= v,
        "lt": lambda s, v: s < v,
        "lte": lambda s, v: s <= v,
    }
    sys.modules[catalogs_name] = fake_catalogs

    return importlib.import_module(module_name)


def test_apply_segmentation_returns_original_dataframe_when_disabled() -> None:
    """Bypass segmentation entirely when config marks segmentation as disabled."""
    segment_module = _import_segment_with_stubbed_catalogs()
    data = pd.DataFrame({"hotel": ["City Hotel"], "adr": [100.0]})
    seg_cfg = SegmentationConfig(enabled=False, include_in_model=False, filters=[])

    result = segment_module.apply_segmentation(data, seg_cfg)

    assert result is data


def test_apply_segmentation_filters_rows_and_drops_segmentation_columns() -> None:
    """Apply configured filters and remove segmentation columns from the output frame."""
    segment_module = _import_segment_with_stubbed_catalogs()
    data = pd.DataFrame(
        {
            "hotel": ["City Hotel", "Resort Hotel", "City Hotel"],
            "market_segment": ["Online TA", "Online TA", "Offline TA/TO"],
            "adr": [110.0, 90.0, 130.0],
        }
    )
    seg_cfg = SegmentationConfig(
        enabled=True,
        include_in_model=False,
        filters=[
            SegmentationFilter(column="hotel", op="eq", value="City Hotel"),
            SegmentationFilter(column="market_segment", op="eq", value="Online TA"),
        ],
    )

    result = segment_module.apply_segmentation(data, seg_cfg)

    assert list(result.columns) == ["adr"]
    assert result["adr"].tolist() == [110.0]


def test_apply_segmentation_raises_for_unsupported_operation() -> None:
    """Raise ``UserError`` when filter operation is not present in the operator registry."""
    segment_module = _import_segment_with_stubbed_catalogs()
    data = pd.DataFrame({"hotel": ["City Hotel"], "adr": [100.0]})
    seg_cfg = cast(
        SegmentationConfig,
        SimpleNamespace(
            enabled=True,
            include_in_model=False,
            filters=[SimpleNamespace(column="hotel", op="contains", value="City")],
        ),
    )

    with pytest.raises(UserError, match="Unsupported segmentation op"):
        segment_module.apply_segmentation(data, seg_cfg)


def test_apply_segmentation_raises_when_filter_column_is_missing() -> None:
    """Raise ``DataError`` when segmentation references a column absent from data."""
    segment_module = _import_segment_with_stubbed_catalogs()
    data = pd.DataFrame({"hotel": ["City Hotel"], "adr": [100.0]})
    seg_cfg = SegmentationConfig(
        enabled=True,
        include_in_model=False,
        filters=[SegmentationFilter(column="country", op="eq", value="PRT")],
    )

    with pytest.raises(DataError, match="Segmentation column country not found"):
        segment_module.apply_segmentation(data, seg_cfg)


def test_apply_segmentation_raises_when_filter_value_is_not_present() -> None:
    """Raise ``DataError`` when segmentation value does not exist in the source column."""
    segment_module = _import_segment_with_stubbed_catalogs()
    data = pd.DataFrame({"hotel": ["City Hotel", "Resort Hotel"], "adr": [100.0, 90.0]})
    seg_cfg = SegmentationConfig(
        enabled=True,
        include_in_model=False,
        filters=[SegmentationFilter(column="hotel", op="eq", value="Business Hotel")],
    )

    with pytest.raises(DataError, match="Segmentation value Business Hotel"):
        segment_module.apply_segmentation(data, seg_cfg)


def test_apply_segmentation_supports_membership_filters() -> None:
    """Apply ``in`` filters against list values and keep rows that match allowed members."""
    segment_module = _import_segment_with_stubbed_catalogs()
    data = pd.DataFrame(
        {
            "hotel": ["City Hotel", "Resort Hotel", "City Hotel"],
            "adr": [110.0, 90.0, 130.0],
        }
    )
    seg_cfg = SegmentationConfig(
        enabled=True,
        include_in_model=False,
        filters=[SegmentationFilter(column="hotel", op="in", value=["City Hotel"])],
    )

    result = segment_module.apply_segmentation(data, seg_cfg)

    assert list(result.columns) == ["adr"]
    assert result["adr"].tolist() == [110.0, 130.0]
