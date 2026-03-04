from ml.search.models.experiment_metadata import ExperimentMetadata
from ml.search.models.search_results import SearchResults
from pydantic import BaseModel


class SearchRecord(BaseModel):
    """Structured data model representing a complete search record, including metadata, configuration, and results."""
    metadata: ExperimentMetadata
    config: dict
    search_results: SearchResults
