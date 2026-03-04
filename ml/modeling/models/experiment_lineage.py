
from ml.modeling.models.feature_lineage import FeatureLineage
from pydantic import BaseModel


class ExperimentLineage(BaseModel):
    feature_lineage: list[FeatureLineage]
    target_column: str
    problem: str
    segment: str
    model_version: str
