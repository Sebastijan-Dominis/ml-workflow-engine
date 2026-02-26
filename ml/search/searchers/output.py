from dataclasses import dataclass
from typing import Any


@dataclass
class SearchOutput:
    search_results: dict[str, Any]
    feature_lineage: list[dict]
    pipeline_hash: str
    scoring_method: str