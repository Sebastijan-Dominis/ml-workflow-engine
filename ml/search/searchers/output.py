from dataclasses import dataclass
from typing import Any

from ml.registry.tabular_splits import AllSplitsInfo


@dataclass
class SearchOutput:
    search_results: dict[str, Any]
    feature_lineage: list[dict]
    pipeline_hash: str
    scoring_method: str
    splits_info: AllSplitsInfo