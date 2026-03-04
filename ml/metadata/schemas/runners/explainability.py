from ml.modeling.models.artifacts import Artifacts
from ml.modeling.models.config_fingerprint import ConfigFingerprint
from ml.modeling.models.experiment_lineage import ExperimentLineage
from ml.modeling.models.run_identity import RunIdentity
from pydantic import BaseModel


class ExplainabilityRunIdentity(RunIdentity):
    """Model representing the identity of an explainability run."""
    stage: str = "explainability"
    explain_run_id: str

class ExplainabilityArtifacts(Artifacts):
    """Model representing the artifacts produced by the explainability runner."""
    top_k_feature_importances_path: str = ""
    top_k_feature_importances_hash: str = ""
    top_k_shap_importances_path: str = ""
    top_k_shap_importances_hash: str = ""

class ExplainabilityMetadata(BaseModel):
    """Metadata for explainability runs."""

    run_identity: ExplainabilityRunIdentity
    lineage: ExperimentLineage
    config_fingerprint: ConfigFingerprint
    artifacts: ExplainabilityArtifacts
    top_k: int
