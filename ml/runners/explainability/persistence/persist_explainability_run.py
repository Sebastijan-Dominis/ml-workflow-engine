"""Persistence orchestration for explainability artifacts and metadata."""

import logging
from pathlib import Path

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.io.persistence.save_metadata import save_metadata
from ml.modeling.models.artifacts import Artifacts
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.modeling.validation.artifacts import validate_explainability_artifacts
from ml.runners.explainability.constants.explainability_metrics_class import ExplainabilityMetrics
from ml.runners.explainability.persistence.save_metrics_csv import save_metrics_csv
from ml.utils.hashing.service import hash_artifact
from ml.utils.runtime.save_runtime import save_runtime_snapshot

logger = logging.getLogger(__name__)

def persist_explainability_run(
        model_cfg: TrainModelConfig,
        *,
        explain_run_id: str,
        train_run_id: str,
        experiment_dir: Path,
        explain_run_dir: Path,
        explainability_metrics: ExplainabilityMetrics,
        feature_lineage: list[FeatureLineage],
        start_time: float,
        timestamp: str,
        artifacts: Artifacts,
        pipeline_cfg_hash: str,
        top_k: int
) -> None:
    """Persist explainability metrics, metadata, and runtime snapshot.

    Args:
        model_cfg: Validated training model configuration.
        explain_run_id: Explainability run identifier.
        train_run_id: Upstream training run identifier.
        experiment_dir: Base experiment directory.
        explain_run_dir: Explainability output directory.
        explainability_metrics: Computed explainability metrics container.
        feature_lineage: Feature lineage records.
        start_time: Process start time used for runtime metadata.
        timestamp: Run timestamp string.
        artifacts: Mutable artifact-path/hash mapping.
        pipeline_cfg_hash: Pipeline configuration hash.
        top_k: Number of top features persisted per explainability method.

    Returns:
        None.
    """

    explainability_artifacts_raw = {
        "model_hash": artifacts.model_hash,
        "model_path": Path(artifacts.model_path).as_posix(),
    }

    if artifacts.pipeline_path and artifacts.pipeline_hash:
        explainability_artifacts_raw["pipeline_path"] = Path(artifacts.pipeline_path).as_posix()
        explainability_artifacts_raw["pipeline_hash"] = artifacts.pipeline_hash

    if explainability_metrics.top_k_feature_importances is not None:
        feature_importances_file = explain_run_dir / "top_k_feature_importances.csv"
        save_metrics_csv(explainability_metrics.top_k_feature_importances, target_file=feature_importances_file, name="Feature importances")
        explainability_artifacts_raw["top_k_feature_importances_path"] = Path(feature_importances_file).as_posix()
        feature_importances_hash = hash_artifact(Path(feature_importances_file))
        explainability_artifacts_raw["top_k_feature_importances_hash"] = feature_importances_hash

    if explainability_metrics.top_k_shap_importances is not None:
        shap_importances_file = explain_run_dir / "top_k_shap_importances.csv"
        save_metrics_csv(explainability_metrics.top_k_shap_importances, target_file=shap_importances_file, name="SHAP importances")
        explainability_artifacts_raw["top_k_shap_importances_path"] = Path(shap_importances_file).as_posix()
        shap_importances_hash = hash_artifact(Path(shap_importances_file))
        explainability_artifacts_raw["top_k_shap_importances_hash"] = shap_importances_hash

    explainability_artifacts = validate_explainability_artifacts(explainability_artifacts_raw)

    metadata = {
        "run_identity": {
            "stage": "explainability",
            "explain_run_id": explain_run_id,
            "train_run_id": train_run_id,
            "snapshot_id": experiment_dir.name,
            "status": "success",
        },
        "lineage": {
            "feature_lineage": [f.model_dump() for f in feature_lineage],
            "target_column": model_cfg.target.name,
            "problem": model_cfg.problem,
            "segment": model_cfg.segment.name,
            "model_version": model_cfg.version,
        },
        "config_fingerprint": {
            "config_hash": model_cfg.meta.config_hash,
            "pipeline_cfg_hash": pipeline_cfg_hash,
        },
        "artifacts": explainability_artifacts.model_dump(exclude_none=True),
        "top_k": top_k
    }

    save_metadata(metadata, target_dir=explain_run_dir)

    save_runtime_snapshot(
        target_dir=explain_run_dir,
        timestamp=timestamp,
        hardware_info=model_cfg.training.hardware,
        start_time=start_time
    )
