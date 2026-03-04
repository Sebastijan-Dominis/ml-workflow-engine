"""Typed output model returned by searcher implementations."""

from dataclasses import dataclass
from typing import Any

from ml.modeling.models.feature_lineage import FeatureLineage
from ml.types import AllSplitsInfo


@dataclass
class SearchOutput:
    """Search results plus lineage, scoring method, and split metadata."""

    search_results: dict[str, Any]
    feature_lineage: list[FeatureLineage]
    pipeline_hash: str
    scoring_method: str
    splits_info: AllSplitsInfo
