
from pydantic import BaseModel

from ml.modeling.models.feature_lineage import FeatureLineage


class ExperimentLineage(BaseModel):
    feature_lineage: list[FeatureLineage]
    target_column: str
    problem: str
    segment: str
    model_version: str
