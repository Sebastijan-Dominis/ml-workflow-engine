"""Persistence orchestration for evaluation run artifacts and metadata."""

import logging
from pathlib import Path

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import PersistenceError
from ml.io.persistence.save_metadata import save_metadata
from ml.modeling.models.artifacts import Artifacts
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.modeling.validation.artifacts import validate_evaluation_artifacts
from ml.runners.evaluation.models.predictions import PredictionArtifacts, PredictionsPathsAndHashes
from ml.runners.evaluation.persistence.prepare_metadata import prepare_metadata
from ml.runners.evaluation.persistence.save_predictions import save_predictions
from ml.runners.shared.persistence.save_metrics import save_metrics
from ml.utils.hashing.service import hash_artifact
from ml.utils.runtime.save_runtime import save_runtime_snapshot

logger = logging.getLogger(__name__)

def persist_evaluation_run(
    model_cfg: TrainModelConfig,
    *,
    eval_run_id: str,
    train_run_id: str,
    experiment_dir: Path,
    eval_run_dir: Path,
    metrics: dict[str, dict[str, float]],
    prediction_dfs: PredictionArtifacts,
    feature_lineage: list[FeatureLineage],
    start_time: float,
    timestamp: str,
    artifacts: Artifacts,
    pipeline_cfg_hash: str
) -> None:
    """Persist evaluation metrics, predictions, metadata, and runtime snapshot.

    Args:
        model_cfg: Validated training model configuration.
        eval_run_id: Evaluation run identifier.
        train_run_id: Upstream training run identifier.
        experiment_dir: Base experiment directory.
        eval_run_dir: Evaluation output directory.
        metrics: Evaluation metrics payload.
        prediction_dfs: Per-split prediction dataframes.
        feature_lineage: Feature lineage records.
        start_time: Process start time used for runtime metadata.
        timestamp: Run timestamp string.
        artifacts: Mutable artifact-path/hash mapping.
        pipeline_cfg_hash: Pipeline configuration hash.

    Returns:
        None.
    """

    metrics_file = save_metrics(
        metrics,
        model_cfg=model_cfg,
        target_run_id=eval_run_id,
        experiment_dir=experiment_dir,
        stage="evaluation"
    )
    evaluation_artifacts_raw = {
        "model_path": Path(artifacts.model_path).as_posix(),
        "model_hash": artifacts.model_hash,
        "metrics_path": Path(metrics_file).as_posix(),
        "metrics_hash": hash_artifact(Path(metrics_file))
    }

    if artifacts.pipeline_hash and artifacts.pipeline_path:
        evaluation_artifacts_raw["pipeline_path"] = Path(artifacts.pipeline_path).as_posix()
        evaluation_artifacts_raw["pipeline_hash"] = artifacts.pipeline_hash

    predictions_paths = save_predictions(prediction_dfs, target_dir=eval_run_dir)

    predictions_paths_and_hashes_raw = {
        "train_predictions_path": predictions_paths.train_predictions_path,
        "val_predictions_path": predictions_paths.val_predictions_path,
        "test_predictions_path": predictions_paths.test_predictions_path,
        "train_predictions_hash": hash_artifact(Path(predictions_paths.train_predictions_path)),
        "val_predictions_hash": hash_artifact(Path(predictions_paths.val_predictions_path)),
        "test_predictions_hash": hash_artifact(Path(predictions_paths.test_predictions_path)),
    }
    try:
        predictions_paths_and_hashes = PredictionsPathsAndHashes(**predictions_paths_and_hashes_raw)
    except Exception as e:
        msg = "Failed to construct predictions paths and hashes model."
        logger.exception(msg)
        raise PersistenceError(msg) from e

    evaluation_artifacts_raw.update(predictions_paths_and_hashes.model_dump())

    evaluation_artifacts = validate_evaluation_artifacts(evaluation_artifacts_raw)

    metadata = prepare_metadata(
        model_cfg=model_cfg,
        eval_run_id=eval_run_id,
        train_run_id=train_run_id,
        experiment_dir=experiment_dir,
        feature_lineage=feature_lineage,
        artifacts=evaluation_artifacts,
        pipeline_cfg_hash=pipeline_cfg_hash
    )

    save_metadata(metadata, target_dir=eval_run_dir)

    save_runtime_snapshot(
        target_dir=eval_run_dir,
        timestamp=timestamp,
        hardware_info=model_cfg.training.hardware,
        start_time=start_time
    )
