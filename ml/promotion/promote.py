import argparse
import copy
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from filelock import FileLock

from ml.cli.error_handling import resolve_exit_code
from ml.exceptions import ConfigError, PersistenceError, UserError
from ml.logging_config import setup_logging
from ml.promotion.config.models import (Direction, MetricName, MetricSet,
                                        PromotionThresholds)
from ml.registry.hash_registry import hash_thresholds
from ml.utils.git import get_git_commit
from ml.utils.loaders import load_json, load_yaml
from ml.utils.persistence.save_metadata import save_metadata
from ml.utils.runtime.runtime_snapshot import (get_conda_env_export,
                                               hash_environment)

logger = logging.getLogger(__name__)

Stage = Literal["staging", "production"]

@dataclass
class RunnersMetadata():
    train_metadata: dict
    eval_metadata: dict
    explain_metadata: dict

@dataclass
class ThresholdComparisonResult():
    meets_thresholds: bool
    message: str
    target_sets: list[MetricSet]
    target_metrics: list[MetricName]
    directions: dict[MetricName, Direction]

@dataclass
class ProductionComparisonResult():
    beats_previous: bool
    message: str
    previous_production_metrics: dict | None

@dataclass
class PreviousProductionRunIdentity():
    experiment_id: str | None
    train_run_id: str | None
    eval_run_id: str | None
    explain_run_id: str | None
    promotion_id: str | None

EPSILON = 1e-8

COMPARISON_DIRECTIONS = {
    Direction.MAXIMIZE: lambda new, old: new > old + EPSILON,
    Direction.MINIMIZE: lambda new, old: new < old - EPSILON
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model.")

    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Model problem, e.g., 'no_show'"
    )

    parser.add_argument(
        "--segment",
        type=str,
        required=True,
        help="Model segment name, e.g., 'city_hotel_online_ta'"
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version, e.g., 'v1'"
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment id (directory name under experiments/{problem}/{segment}/{version})"
    )

    parser.add_argument(
        "--train-run-id",
        type=str,
        required=True,
        help="Train run id (directory name under experiments/{problem}/{segment}/{version}/{experiment_id}/training)"
    )

    parser.add_argument(
        "--eval-run-id",
        type=str,
        required=True,
        help="Eval run id (directory name under experiments/{problem}/{segment}/{version}/{experiment_id}/evaluation)"
    )

    parser.add_argument(
        "--explain-run-id",
        type=str,
        required=True,
        help="Explain run id (directory name under experiments/{problem}/{segment}/{version}/{experiment_id}/explainability)"
    )

    parser.add_argument(
        "--stage",
        choices=["staging", "production"],
        required=True,
        help="Stage of the promotion (staging or production)"
    )

    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    return parser.parse_args()

def validate_run_dirs(train_run_dir: Path, eval_run_dir: Path, explain_run_dir: Path) -> None:
    if not train_run_dir.exists():
        msg = f"Train run directory does not exist: {train_run_dir}"
        logger.error(msg)
        raise UserError(msg)
    if not eval_run_dir.exists():
        msg = f"Eval run directory does not exist: {eval_run_dir}"
        logger.error(msg)
        raise UserError(msg)
    if not explain_run_dir.exists():
        msg = f"Explain run directory does not exist: {explain_run_dir}"
        logger.error(msg)
        raise UserError(msg)

def get_runners_metadata(train_run_dir: Path, eval_run_dir: Path, explain_run_dir: Path) -> RunnersMetadata:
    train_metadata = load_json(train_run_dir / "metadata.json")
    eval_metadata = load_json(eval_run_dir / "metadata.json")
    explain_metadata = load_json(explain_run_dir / "metadata.json")
    return RunnersMetadata(train_metadata, eval_metadata, explain_metadata)

def validate_run_ids(
    *,
    args: argparse.Namespace,
    runners_metadata: RunnersMetadata
) -> None:
    train_metadata = runners_metadata.train_metadata
    eval_metadata = runners_metadata.eval_metadata
    explain_metadata = runners_metadata.explain_metadata

    if eval_metadata.get("run_identity", {}).get("train_run_id") != train_metadata.get("run_identity", {}).get("train_run_id"):
        msg = f"Evaluation run {args.eval_run_id} is not linked to train run {args.train_run_id}"
        logger.error(msg)
        raise UserError(msg)
    
    if explain_metadata.get("run_identity", {}).get("train_run_id") != train_metadata.get("run_identity", {}).get("train_run_id"):
        msg = f"Explain run {args.explain_run_id} is not linked to train run {args.train_run_id}"
        logger.error(msg)
        raise UserError(msg)

def validate_explainability_artifacts(runners_metadata: RunnersMetadata, args: argparse.Namespace) -> None:
    explain_status = runners_metadata.explain_metadata.get("run_identity", {}).get("status")
    if explain_status != "success":
        msg = f"Explain run {args.explain_run_id} did not complete successfully. Status: {explain_status}"
        logger.error(msg)
        raise UserError(msg)

    train_artifacts = runners_metadata.train_metadata.get("artifacts", {})
    explain_artifacts = runners_metadata.explain_metadata.get("artifacts", {})

    if explain_artifacts.get("model_hash") is None:
        msg = f"Explain run {args.explain_run_id} is missing model hash artifact."
        logger.error(msg)
        raise UserError(msg)
    if explain_artifacts.get("model_hash") != train_artifacts.get("model_hash"):
        msg = f"Model hash in explain run {args.explain_run_id} does not match model hash in train run {args.train_run_id}."
        logger.error(msg)
        raise UserError(msg)
    
    if explain_artifacts.get("pipeline_hash") is None:
        msg = f"Explain run {args.explain_run_id} is missing pipeline hash artifact."
        logger.error(msg)
        raise UserError(msg)
    if explain_artifacts.get("pipeline_hash") != train_artifacts.get("pipeline_hash"):
        msg = f"Pipeline hash in explain run {args.explain_run_id} does not match pipeline hash in train run {args.train_run_id}."
        logger.error(msg)
        raise UserError(msg)

def extract_thresholds(promotion_thresholds: dict, problem: str, segment: str) -> dict:
    promotion_thresholds = promotion_thresholds.get(problem, {}).get(segment, {})
    if not promotion_thresholds:
        msg = f"No promotion thresholds found for problem={problem} segment={segment}"
        logger.error(msg)
        raise UserError(msg)
    
    return promotion_thresholds

def validate_promotion_thresholds(promotion_thresholds: dict) -> PromotionThresholds:
    try:
        return PromotionThresholds(**promotion_thresholds)
    except Exception as e:
        msg = f"Invalid promotion thresholds configuration. Configuration: {promotion_thresholds}"
        logger.exception(msg)
        raise ConfigError(msg) from e

def compare_against_thresholds(
    *,
    evaluation_metrics: dict[str, dict[str, float]], 
    promotion_thresholds: PromotionThresholds
) -> ThresholdComparisonResult:
    target_sets = promotion_thresholds.promotion_metrics.sets
    target_metrics = promotion_thresholds.promotion_metrics.metrics
    directions = promotion_thresholds.promotion_metrics.directions

    for metric_set in target_sets:
        thresholds_for_set = promotion_thresholds.thresholds.model_dump().get(metric_set)
        if thresholds_for_set is None:
            msg = f"Thresholds for metric set '{metric_set}' are not defined."
            logger.error(msg)
            raise ConfigError(msg)
        
        for metric in target_metrics:
            threshold_value = thresholds_for_set.get(metric)
            if threshold_value is None:
                msg = f"Threshold value for metric '{metric}' in set '{metric_set}' is not defined."
                logger.error(msg)
                raise ConfigError(msg)
            
            metric_value = evaluation_metrics.get(metric_set, {}).get(metric)
            if metric_value is None:
                msg = f"Evaluation metric '{metric}' is not available in the evaluation metrics."
                logger.error(msg)
                raise UserError(msg)

            direction = directions.get(metric)
            if direction is None:
                msg = f"Direction for metric '{metric}' is not defined."
                logger.error(msg)
                raise ConfigError(msg)
            
            comparison_func = COMPARISON_DIRECTIONS.get(direction)
            if not comparison_func:
                msg = f"Invalid direction '{direction}' for metric '{metric}'."
                logger.error(msg)
                raise ConfigError(msg)
            
            if not comparison_func(metric_value, threshold_value):
                msg = f"Metric '{metric}' with value {metric_value} does not meet the promotion threshold of {threshold_value} in set '{metric_set}'."
                logger.warning(f"Promotion criteria not met: {msg}")
                return ThresholdComparisonResult(
                    meets_thresholds=False,
                    message=msg,
                    target_sets=target_sets,
                    target_metrics=target_metrics,
                    directions=directions
                )
            else:
                logger.debug(f"Metric '{metric}' with value {metric_value} meets the promotion threshold of {threshold_value} in set '{metric_set}'.")
    return ThresholdComparisonResult(
        meets_thresholds=True,
        message="All promotion criteria regarding thresholds met.",
        target_sets=target_sets,
        target_metrics=target_metrics,
        directions=directions
    )

def compare_against_production_model(
    *,
    evaluation_metrics: dict[str, dict[str, float]], 
    current_prod_model_info: dict,
    metric_sets: list[MetricSet],
    metric_names: list[MetricName],
    directions: dict[MetricName, Direction]
) -> ProductionComparisonResult:
    if not current_prod_model_info:
        msg = "No current production model found. Skipping comparison against production model."
        logger.warning(msg)
        return ProductionComparisonResult(
            beats_previous=True,
            message=msg,
            previous_production_metrics=None
        )
    
    prod_metrics = current_prod_model_info.get("metrics", {})
    if not prod_metrics:
        msg = "Current production model does not have metrics information."
        logger.error(msg)
        raise UserError(msg)
    
    for metric_set in metric_sets:
        for metric in metric_names:
            prod_metric_value = prod_metrics.get(metric_set, {}).get(metric)
            if prod_metric_value is None:
                msg = f"Production model is missing metric '{metric}' in set '{metric_set}'. Cannot compare against production model."
                logger.error(msg)
                raise UserError(msg)

            eval_metric_value = evaluation_metrics.get(metric_set, {}).get(metric)
            if eval_metric_value is None:
                msg = f"Evaluation metrics are missing metric '{metric}' in set '{metric_set}'. Cannot compare against production model."
                logger.error(msg)
                raise UserError(msg)

            direction = directions.get(metric)
            if direction is None:
                msg = f"Direction for metric '{metric}' is not defined."
                logger.error(msg)
                raise ConfigError(msg)
            comparison_func = COMPARISON_DIRECTIONS.get(direction)
            if not comparison_func:
                msg = f"Invalid direction '{direction}' for metric '{metric}'."
                logger.error(msg)
                raise ConfigError(msg)
            
            if not comparison_func(eval_metric_value, prod_metric_value):
                msg = f"Metric '{metric}' in set '{metric_set}' does not outperform production model. Evaluation value: {eval_metric_value}, Production value: {prod_metric_value}."
                logger.warning(f"Promotion criteria not met: {msg}")
                return ProductionComparisonResult(
                    beats_previous=False,
                    message=msg,
                    previous_production_metrics=prod_metrics
                )
            else:
                logger.debug(f"Metric '{metric}' in set '{metric_set}' outperforms production model. Evaluation value: {eval_metric_value}, Production value: {prod_metric_value}.")
    return ProductionComparisonResult(
        beats_previous=True,
        message="Model outperforms production model on all metrics.",
        previous_production_metrics=prod_metrics
    )

def get_artifacts(explain_metadata: dict) -> dict:
    artifacts = explain_metadata.get("artifacts", {})

    if not artifacts or artifacts.get("model_hash") is None or artifacts.get("model_path") is None:
        msg = f"Explainability metadata is missing required artifact information. Artifacts found: {artifacts}"
        logger.error(msg)
        raise PersistenceError(msg)
    
    return artifacts

def get_feature_lineage(training_metadata: dict) -> list[str]:
    feature_lineage = training_metadata.get("lineage", {}).get("feature_lineage")
    if not feature_lineage:
        msg = "Training metadata is missing feature lineage information."
        logger.error(msg)
        raise PersistenceError(msg)
    return feature_lineage

def get_pipeline_cfg_hash(training_metadata: dict) -> str:
    pipeline_cfg_hash = training_metadata.get("config_fingerprint", {}).get("pipeline_cfg_hash")
    if not pipeline_cfg_hash:
        msg = "Training metadata is missing pipeline configuration hash information."
        logger.error(msg)
        raise PersistenceError(msg)
    return pipeline_cfg_hash

def prepare_run_information(
    *,
    args: argparse.Namespace,
    experiment_id: str, 
    train_run_id: str, 
    eval_run_id: str, 
    explain_run_id: str, 
    stage: Stage,
    run_id: str | None,
    timestamp: str,
    training_metadata: dict,
    explain_metadata: dict,
    metrics: dict, 
    git_commit: str,
) -> dict:
    artifacts = get_artifacts(explain_metadata)

    feature_lineage = get_feature_lineage(training_metadata)

    pipeline_cfg_hash = get_pipeline_cfg_hash(training_metadata)

    run_info = {
        "experiment_id": experiment_id,
        "train_run_id": train_run_id,
        "eval_run_id": eval_run_id,
        "explain_run_id": explain_run_id,
        "model_version": args.version,
        "pipeline_cfg_hash": pipeline_cfg_hash,

        "artifacts": artifacts,

        "feature_lineage": feature_lineage,

        "metrics": metrics,

        "git_commit": git_commit,
    }

    if stage == "production":
        run_info["promotion_id"] = run_id
        run_info["promoted_at"] = timestamp    
    elif stage == "staging":
        run_info["staging_id"] = run_id
        run_info["staged_at"] = timestamp
    
    return run_info

def update_registry_and_archive(
    *,
    model_registry: dict, 
    archive_registry: dict,
    stage: Stage,
    run_info: dict,
    problem: str,
    segment: str,
    registry_path: Path = Path("model_registry") / "models.yaml",
    archive_path: Path = Path("model_registry") / "archive.yaml"
) -> dict:
    new_registry = copy.deepcopy(model_registry, {})
    
    new_registry.setdefault(problem, {})
    new_registry[problem].setdefault(segment, {})
    new_archive = None

    if stage == "production":
        current_prod_model_info = new_registry[problem][segment].get("production")
        if current_prod_model_info:
            promotion_id = current_prod_model_info.get("promotion_id")
            if not promotion_id:
                msg = f"Current production model for problem '{problem}' and segment '{segment}' is missing promotion_id. Cannot archive without promotion_id. Current production model info: {current_prod_model_info}"
                logger.error(msg)
                raise PersistenceError(msg)
            new_archive = copy.deepcopy(archive_registry, {})
            new_archive.setdefault(problem, {}).setdefault(segment, {})
            new_archive[problem][segment][promotion_id] = current_prod_model_info
        
        new_registry[problem][segment]["production"] = run_info
    else:
        # Only one staging model is allowed per problem/segment, so we can directly set it without archiving
        new_registry[problem][segment]["staging"] = run_info

    try:
        if new_archive is not None:
            temp_archive_path = archive_path.with_suffix(".tmp")
            with open(temp_archive_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(new_archive, f, sort_keys=False)
            os.replace(temp_archive_path, archive_path)

        temp_registry_path = registry_path.with_suffix(".tmp")
        with open(temp_registry_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(new_registry, f, sort_keys=False)
        os.replace(temp_registry_path, registry_path)

        return new_registry
    except Exception as e:
        msg = f"Failed to update model registry and archive. Run info: {run_info}"
        logger.exception(msg)
        raise PersistenceError(msg) from e

def persist_registry_diff(
    *,
    previous_registry: dict,
    updated_registry: dict,
    run_dir: Path
) -> None:
    diff_path = run_dir / "registry_diff.yaml"
    diff = {
        "previous": previous_registry,
        "updated": updated_registry
    }
    try:
        with open(diff_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(diff, f, sort_keys=False)
    except Exception as e:
        msg = f"Failed to persist registry diff to {diff_path}"
        logger.exception(msg)
        raise PersistenceError(msg) from e

def get_training_conda_env_hash(train_run_dir: Path) -> str:
    training_runtime_file = train_run_dir / "runtime.json"
    training_runtime = load_json(training_runtime_file)
    training_conda_env_hash = training_runtime.get("environment", {}).get("conda_env_hash")
    if not training_conda_env_hash:
        msg = f"Training runtime information is missing conda environment hash. Runtime file: {training_runtime_file}"
        logger.error(msg)
        raise PersistenceError(msg)
    return training_conda_env_hash

def prepare_metadata(
    *,
    run_id: str | None,
    stage: Stage,
    args: argparse.Namespace,
    metrics: dict,
    previous_production_metrics: dict | None,
    promotion_thresholds: PromotionThresholds, 
    promoted: bool,
    beats_previous: bool,
    reason: str,
    git_commit: str,
    timestamp: str,
    previous_production_run_identity: PreviousProductionRunIdentity,
    train_run_dir: Path
) -> dict:
    conda_env_export = get_conda_env_export()
    promotion_conda_env_hash = hash_environment(conda_env_export)

    training_conda_env_hash = get_training_conda_env_hash(train_run_dir)

    if promotion_conda_env_hash != training_conda_env_hash:
        msg = f"Conda environment hash for promotion process does not match conda environment hash for training run. Promotion conda env hash: {promotion_conda_env_hash}, Training conda env hash: {training_conda_env_hash}. This may indicate that the promotion process is running with a different conda environment than the training process, which could lead to inconsistencies and unexpected issues. Please ensure that the same conda environment is used for both training and promotion processes."
        logger.warning(msg)

    thresholds_hash = hash_thresholds(promotion_thresholds.model_dump())

    metadata = {
        "run_identity": {
            "experiment_id": args.experiment_id,
            "train_run_id": args.train_run_id,
            "eval_run_id": args.eval_run_id,
            "explain_run_id": args.explain_run_id,
            "stage": stage,
        },

        "previous_production_run_identity": {
            "experiment_id": previous_production_run_identity.experiment_id,
            "train_run_id": previous_production_run_identity.train_run_id,
            "eval_run_id": previous_production_run_identity.eval_run_id,
            "explain_run_id": previous_production_run_identity.explain_run_id,
            "promotion_id": previous_production_run_identity.promotion_id,
        },

        "metrics": metrics,

        "previous_production_metrics": previous_production_metrics,

        "promotion_thresholds": promotion_thresholds.model_dump(),
        "promotion_thresholds_hash": thresholds_hash,

        "decision": {
            "promoted": promoted,
            "reason": reason,
            "beats_previous": beats_previous
        },

        "context": {
            "git_commit": git_commit,
            "promotion_conda_env_hash": promotion_conda_env_hash,
            "training_conda_env_hash": training_conda_env_hash,
            "timestamp": timestamp
        }
    }

    if stage == "production":
        metadata["run_identity"]["promotion_id"] = run_id
    elif stage == "staging":
        metadata["run_identity"]["staging_id"] = run_id

    return metadata
    
def main() -> int:
    args = parse_args()

    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"

    model_registry_dir = Path("model_registry")
    run_dir = model_registry_dir / "runs" / run_id
    promotion_configs_dir = Path("configs") / "promotion"
    train_run_dir = Path("experiments") / args.problem / args.segment / args.version / args.experiment_id / "training" / args.train_run_id
    eval_run_dir = Path("experiments") / args.problem / args.segment / args.version / args.experiment_id / "evaluation" / args.eval_run_id
    explain_run_dir = Path("experiments") / args.problem / args.segment / args.version / args.experiment_id / "explainability" / args.explain_run_id

    run_dir.mkdir(parents=True, exist_ok=False)

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)

    log_file = run_dir / "promotion.log"
    setup_logging(log_file, log_level)

    try:
        validate_run_dirs(train_run_dir, eval_run_dir, explain_run_dir)
        runners_metadata = get_runners_metadata(train_run_dir, eval_run_dir, explain_run_dir)
        validate_run_ids(args=args, runners_metadata=runners_metadata)
        validate_explainability_artifacts(runners_metadata=runners_metadata, args=args)

        registry_path = model_registry_dir / "models.yaml"
        lock = FileLock(str(registry_path) + ".lock", timeout=300)  # 5 minute timeout for acquiring the lock

        with lock:
            model_registry = load_yaml(model_registry_dir / "models.yaml")
            archive_registry = load_yaml(model_registry_dir / "archive.yaml")
            global_promotion_thresholds = load_yaml(promotion_configs_dir / "thresholds.yaml")
            evaluation_metrics_file = load_json(eval_run_dir / "metrics.json")
            evaluation_metrics = evaluation_metrics_file.get("metrics", {})

            promotion_thresholds_raw = extract_thresholds(global_promotion_thresholds, args.problem, args.segment)
            promotion_thresholds = validate_promotion_thresholds(promotion_thresholds_raw)

            current_prod_model_info = (
                model_registry
                .get(args.problem, {})
                .get(args.segment, {})
                .get("production")
            )

            git_commit = get_git_commit()

            previous_production_run_identity = PreviousProductionRunIdentity(
                experiment_id=current_prod_model_info.get("experiment_id") if current_prod_model_info else None,
                train_run_id=current_prod_model_info.get("train_run_id") if current_prod_model_info else None,
                eval_run_id=current_prod_model_info.get("eval_run_id") if current_prod_model_info else None,
                explain_run_id=current_prod_model_info.get("explain_run_id") if current_prod_model_info else None,
                promotion_id=current_prod_model_info.get("promotion_id") if current_prod_model_info else None
            )

            threshold_comparison = compare_against_thresholds(
                evaluation_metrics=evaluation_metrics,
                promotion_thresholds=promotion_thresholds
            )

            updated_registry = None

            if args.stage == "production":
                production_comparison = compare_against_production_model(
                    evaluation_metrics=evaluation_metrics, current_prod_model_info=current_prod_model_info, 
                    metric_sets=threshold_comparison.target_sets, 
                    metric_names=threshold_comparison.target_metrics,
                    directions=threshold_comparison.directions
                )

                promotion_decision = threshold_comparison.meets_thresholds and production_comparison.beats_previous

                if promotion_decision:
                    run_info = prepare_run_information(
                        args=args,
                        experiment_id=args.experiment_id,
                        train_run_id=args.train_run_id,
                        eval_run_id=args.eval_run_id,
                        explain_run_id=args.explain_run_id,
                        stage=args.stage,
                        run_id=run_id,
                        timestamp=timestamp,
                        explain_metadata=runners_metadata.explain_metadata,
                        training_metadata=runners_metadata.train_metadata,
                        metrics=evaluation_metrics,
                        git_commit=git_commit
                    )
                    updated_registry = update_registry_and_archive(
                        model_registry=model_registry,
                        archive_registry=archive_registry,
                        run_info=run_info,
                        stage=args.stage,
                        problem=args.problem,
                        segment=args.segment,
                        registry_path=model_registry_dir / "models.yaml",
                        archive_path=model_registry_dir / "archive.yaml"
                    )
                    reason = "Model meets all promotion criteria."

                    previous_id = (
                        current_prod_model_info.get("promotion_id")
                        if current_prod_model_info
                        else None
                    )
                    logger.info(
                        "Model promoted and previous production model with promotion_id '%s' archived successfully.", previous_id
                    )
                else:
                    reasons = []
                    if not threshold_comparison.meets_thresholds:
                        reasons.append(threshold_comparison.message)
                    if not production_comparison.beats_previous:
                        reasons.append(production_comparison.message)
                    reason = "; ".join(reasons)
                    logger.info(f"Model promotion criteria not met. Reasoning: {reason}")

                metadata = prepare_metadata(
                    run_id=run_id,
                    stage=args.stage,
                    args=args,
                    metrics=evaluation_metrics,
                    previous_production_metrics=production_comparison.previous_production_metrics,
                    promotion_thresholds=promotion_thresholds,
                    promoted=promotion_decision,
                    beats_previous=production_comparison.beats_previous,
                    reason=reason,
                    git_commit=git_commit,
                    timestamp=timestamp,
                    previous_production_run_identity=previous_production_run_identity,
                    train_run_dir=train_run_dir
                )

            elif args.stage == "staging":
                promotion_decision = threshold_comparison.meets_thresholds

                if promotion_decision:
                    run_info = prepare_run_information(
                        args=args,
                        experiment_id=args.experiment_id,
                        train_run_id=args.train_run_id,
                        eval_run_id=args.eval_run_id,
                        explain_run_id=args.explain_run_id,
                        stage=args.stage,
                        run_id=run_id,
                        timestamp=timestamp,
                        explain_metadata=runners_metadata.explain_metadata,
                        training_metadata=runners_metadata.train_metadata,
                        metrics=evaluation_metrics,
                        git_commit=git_commit
                    )
                    updated_registry = update_registry_and_archive(
                        model_registry=model_registry,
                        archive_registry=archive_registry,
                        run_info=run_info,
                        stage=args.stage,
                        problem=args.problem,
                        segment=args.segment,
                        registry_path=model_registry_dir / "models.yaml",
                        archive_path=model_registry_dir / "archive.yaml"
                    )
                    reason = "Model promoted to staging. No comparison against production model for staging promotion."

                    logger.info("Model promoted to staging successfully.")
                else:
                    reason = threshold_comparison.message
                    logger.info(f"Model staging criteria not met. Reasoning: {reason}")

                metadata = prepare_metadata(
                    run_id=run_id,
                    stage=args.stage,
                    args=args,
                    metrics=evaluation_metrics,
                    previous_production_metrics=None,
                    promotion_thresholds=promotion_thresholds, 
                    promoted=promotion_decision,
                    beats_previous=False,
                    reason=reason,
                    git_commit=git_commit,
                    timestamp=timestamp,
                    previous_production_run_identity=previous_production_run_identity,
                    train_run_dir=train_run_dir
                )
            else:
                msg = f"Invalid promotion stage: {args.stage}"
                logger.error(msg)
                raise UserError(msg)
            
            if updated_registry is not None:
                persist_registry_diff(
                    previous_registry=model_registry,
                    updated_registry=updated_registry,
                    run_dir=run_dir
                )

            save_metadata(metadata, target_dir=run_dir)

            return 0

    except Exception as e:
        logger.exception("An error occurred during promotion.")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())