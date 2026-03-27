import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from ml.cli.error_handling import resolve_exit_code
from ml.exceptions import ConfigError, MonitoringError, PipelineContractError, RuntimeMLError
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.persistence.save_metadata import save_metadata
from ml.logging_config import setup_logging
from ml.modeling.models.metrics import TrainingMetrics
from ml.modeling.validation.metrics import validate_training_metrics
from ml.post_promotion.shared.loading.features import prepare_features
from ml.post_promotion.shared.loading.model_registry import get_model_registry_info
from ml.promotion.config.promotion_thresholds import MetricName, PromotionMetricsConfig
from ml.promotion.config.registry_entry import RegistryEntry
from ml.promotion.validation.promotion_thresholds import validate_promotion_thresholds
from ml.runners.evaluation.evaluators.classification.metrics import (
    compute_metrics as compute_classification_metrics,
)
from ml.runners.evaluation.evaluators.regression.metrics import (
    compute_metrics as compute_regression_metrics,
)
from ml.types.latest import LatestSnapshot
from ml.utils.loaders import load_json, load_yaml
from ml.utils.snapshots.snapshot_path import get_snapshot_path

from pipelines.post_promotion.infer import validate_inference_metadata

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for production and staging models with monitoring-ready outputs.")

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
        "--inference-run-id",
        type=str,
        default=LatestSnapshot.LATEST.value,
        help="Inference run id (directory name under inference_runs/{problem}/{segment}); if not provided, defaults to 'latest' which picks the most recent inference run directory"
    )

    parser.add_argument(
        "--logging-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    return parser.parse_args()

# -----------------------------
# Drift functions
# -----------------------------
def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0  # or raise, depending on your philosophy

    if pd.api.types.is_numeric_dtype(expected):
        bin_edges = np.histogram_bin_edges(expected, bins=bins)
        bin_edges = bin_edges.tolist()  # Convert numpy array to list for pd.cut

        expected_bins = pd.cut(expected, bins=bin_edges, include_lowest=True)
        actual_bins = pd.cut(actual, bins=bin_edges, include_lowest=True)
    else:
        expected_bins = expected.astype(str)
        actual_bins = actual.astype(str)

    e_dist = expected_bins.value_counts(normalize=True)
    a_dist = actual_bins.value_counts(normalize=True)

    all_bins = set(e_dist.index).union(a_dist.index)

    psi_val = 0.0
    for b in all_bins:
        e = e_dist.get(b, 1e-6)
        a = a_dist.get(b, 1e-6)

        # prevent log(0)
        if e == 0:
            e = 1e-6
        if a == 0:
            a = 1e-6

        psi_val += (e - a) * np.log(e / a)

    return float(psi_val)

def compute_ks(expected: pd.Series, actual: pd.Series) -> float:
    """Kolmogorov-Smirnov statistic for numeric features."""
    from scipy.stats import ks_2samp
    result: Any = ks_2samp(expected, actual)
    stat = getattr(result, "statistic", None)
    if stat is None:
        # scipy may return a tuple (statistic, pvalue) in some versions
        stat = result[0]
    return float(stat)

def infer_drift_method(series: pd.Series) -> Literal["ks", "psi"]:
    """
    Decide which drift metric to use for a feature.

    Returns:
        "ks" or "psi"
    """
    if pd.api.types.is_numeric_dtype(series):
        # Low cardinality numeric behaves like categorical
        if series.nunique(dropna=True) < 20:
            return "psi"
        return "ks"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "psi"  # always bin → PSI

    # everything else → categorical
    return "psi"

def analyze_ks_result(feature_name: str, ks_stat: float) -> None:
    if ks_stat < 0 or ks_stat > 1:
        msg = f"KS statistic out of bounds [0,1]: {ks_stat}"
        logger.error(msg)
        raise RuntimeMLError(msg)
    if ks_stat < 0.1:
        logger.debug(f"KS statistic {ks_stat:.4f} indicates low drift for feature: {feature_name}")
    elif ks_stat < 0.25:
        logger.warning(f"KS statistic {ks_stat:.4f} indicates moderate drift for feature: {feature_name}")
    elif ks_stat < 0.5:
        logger.error(f"KS statistic {ks_stat:.4f} indicates high drift for feature: {feature_name}")
    else:
        logger.critical(f"KS statistic {ks_stat:.4f} indicates severe drift for feature: {feature_name}")

def analyze_psi_result(feature_name: str, psi_val: float) -> None:
    if psi_val < 0 or not np.isfinite(psi_val):
        msg = f"PSI value is invalid (negative or non-finite): {psi_val}"
        logger.error(msg)
        raise RuntimeMLError(msg)
    if psi_val < 0.1:
        logger.debug(f"PSI value {psi_val:.4f} indicates low drift for feature: {feature_name}")
    elif psi_val < 0.25:
        logger.warning(f"PSI value {psi_val:.4f} indicates moderate drift for feature: {feature_name}")
    elif psi_val < 0.5:
        logger.error(f"PSI value {psi_val:.4f} indicates high drift for feature: {feature_name}")
    else:
        logger.critical(f"PSI value {psi_val:.4f} indicates severe drift for feature: {feature_name}")

def compute_drift(
    expected: pd.Series,
    actual: pd.Series
) -> float:
    method = infer_drift_method(expected)

    if method == "ks":
        try:
            result = compute_ks(expected, actual)
        except Exception as e:
            msg = f"Error computing KS statistic for feature: {expected.name}"
            logger.exception(msg)
            raise MonitoringError(msg) from e
        feature_name = str(expected.name) if expected.name is not None else "<unknown>"
        analyze_ks_result(feature_name, result)
    elif method == "psi":
        try:
            result = compute_psi(expected, actual)
        except Exception as e:
            msg = f"Error computing PSI for feature: {expected.name}"
            logger.exception(msg)
            raise MonitoringError(msg) from e
        feature_name = str(expected.name) if expected.name is not None else "<unknown>"
        analyze_psi_result(feature_name, result)
    else:
        msg = f"Unsupported drift method: {method}"
        logger.error(msg)
        raise RuntimeMLError(msg)
    return result

@dataclass
class InferenceFeaturesAndTarget:
    features: pd.DataFrame
    target: pd.Series

def load_inference_features_and_target(
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"]
) -> InferenceFeaturesAndTarget:
    inference_dir = Path("predictions") / args.problem / args.segment
    snapshot_path = get_snapshot_path(args.inference_run_id, inference_dir)
    inference_run_dir = snapshot_path / stage

    inference_metadata_path = inference_run_dir / "metadata.json"
    inference_metadata_raw = load_json(inference_metadata_path)
    inference_metadata = validate_inference_metadata(inference_metadata_raw)

    # Assumes supervised inference. Modify as needed for unsupervised tasks.
    inference_features_return = prepare_features(
        args=args,
        model_metadata=model_metadata,
        snapshot_bindings_id=inference_metadata.snapshot_bindings_id
    )
    inference_features = inference_features_return.features

    result = InferenceFeaturesAndTarget(
        features=inference_features,
        target=inference_features_return.target
    )
    return result

def load_training_features(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
) -> pd.DataFrame:
    training_features_return = prepare_features(
        args=args,
        model_metadata=model_metadata
    )
    training_features = training_features_return.features
    return training_features

def compare_feature_distributions(
    *,
    stage: Literal["production", "staging"],
    inference_features: pd.DataFrame,
    training_features: pd.DataFrame
):
    logger.info(f"Comparing feature distributions for the {stage} model...")

    drift_results = {}
    for col in training_features.columns:
        expected = training_features[col]
        actual = inference_features[col]
        drift_score = compute_drift(expected, actual)
        drift_results[col] = drift_score

    return drift_results

def get_promotion_metrics_info(
    args: argparse.Namespace,
) -> PromotionMetricsConfig:
    global_thresholds = load_yaml(Path("configs") / "promotion" / "thresholds.yaml")
    model_thresholds_raw = global_thresholds.get(args.problem, {}).get(args.segment, {})
    if not model_thresholds_raw:
        msg = f"No promotion thresholds found for problem='{args.problem}' and segment='{args.segment}' in thresholds.yaml. File content: {global_thresholds}"
        logger.error(msg)
        raise PipelineContractError(msg)
    model_thresholds = validate_promotion_thresholds(model_thresholds_raw)
    return model_thresholds.promotion_metrics

def get_expected_performance(
    model_metadata: RegistryEntry,
    promotion_metrics_info: PromotionMetricsConfig
):
    expected_performance = {}

    for metric in promotion_metrics_info.metrics:
        curr_metric_value = model_metadata.metrics.test.get(metric, None)

        if not isinstance(curr_metric_value, (int, float)):
            msg = f"Model metadata has null expected value for metric '{metric}' in the test set. Cannot retrieve expected performance for monitoring comparison."
            logger.error(msg)
            raise PipelineContractError(msg)

        expected_performance[metric] = curr_metric_value

    return expected_performance

def load_training_metrics_file(
    args: argparse.Namespace,
    model_metadata: RegistryEntry
) -> TrainingMetrics:
    training_metrics_path = Path("experiments") / args.problem / args.segment / model_metadata.model_version / model_metadata.experiment_id / "training" / model_metadata.train_run_id / "metrics.json"
    training_metrics_raw = load_json(training_metrics_path)
    training_metrics = validate_training_metrics(training_metrics_raw)
    return training_metrics

def load_predictions(
    args: argparse.Namespace,
    stage: Literal["production", "staging"]
) -> pd.DataFrame:
    inference_dir = Path("predictions") / args.problem / args.segment
    snapshot_path = get_snapshot_path(args.inference_run_id, inference_dir)
    inference_run_dir = snapshot_path / stage
    inference_predictions_path = inference_run_dir / "predictions.parquet"
    predictions = pd.read_parquet(inference_predictions_path)
    return predictions

def calculate_current_performance(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"],
    target: pd.Series
):
    predictions = load_predictions(args, stage)
    df = predictions.join(target.rename("target"), how="inner")
    y_true = df["target"]
    y_pred = df["prediction"]
    training_metrics = load_training_metrics_file(args, model_metadata)
    if training_metrics.task_type == "classification":
        threshold = None
        threshold_info = training_metrics.metrics.get("threshold", {})
        if isinstance(threshold_info, dict):
            threshold = threshold_info.get("value")
        if threshold is None or not isinstance(threshold, (int, float)):
            msg = "Training metrics for classification task is missing 'threshold' information. Defaulting to 0.5."
            logger.critical(msg)
            threshold = 0.5
        # Consider changing for multiclass in the future
        y_prob = predictions.get("proba_1") if "proba_1" in predictions.columns else None
        current_performance = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            threshold=threshold,
        )
    elif training_metrics.task_type == "regression":
        current_performance = compute_regression_metrics(
            y_true=y_true,
            y_pred=y_pred
        )
    else:
        msg = f"Unsupported task type in training metrics: {training_metrics.task_type}"
        logger.error(msg)
        raise PipelineContractError(msg)
    return current_performance


def assess_model_performance(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"],
    target: pd.Series,
    promotion_metrics_info: PromotionMetricsConfig
) -> dict[str | MetricName, dict[str, Any]]:
    expected_performance = get_expected_performance(model_metadata, promotion_metrics_info)
    current_performance = calculate_current_performance(
        args=args,
        model_metadata=model_metadata,
        stage=stage,
        target=target
    )

    performance_results: dict[str | MetricName, dict[str, Any]] = {}

    for metric in promotion_metrics_info.metrics:
        expected_value = expected_performance[metric]

        current_value = current_performance[metric]
        if current_value is None or not isinstance(current_value, (int, float)):
            msg = f"Current performance is missing value for metric '{metric}'. Cannot assess model performance against expected thresholds. Current performance content: {current_performance}"
            logger.error(msg)
            raise PipelineContractError(msg)

        direction = promotion_metrics_info.directions[metric]
        if direction == "maximize":
            if current_value < expected_value:
                logger.warning(f"{stage.capitalize()} model performance degradation detected for metric '{metric}': expected >= {expected_value:.4f}, got {current_value:.4f}")
                performance_results[metric] = {
                    "status": "degradation",
                    "expected": expected_value,
                    "current": current_value,
                    "direction": direction
                }
            else:
                logger.info(f"{stage.capitalize()} model performance for metric '{metric}' is acceptable: expected >= {expected_value:.4f}, got {current_value:.4f}")
                performance_results[metric] = {
                    "status": "acceptable",
                    "expected": expected_value,
                    "current": current_value,
                    "direction": direction
                }
        elif direction == "minimize":
            if current_value > expected_value:
                logger.warning(f"{stage.capitalize()} model performance degradation detected for metric '{metric}': expected <= {expected_value:.4f}, got {current_value:.4f}")
                performance_results[metric] = {
                    "status": "degradation",
                    "expected": expected_value,
                    "current": current_value,
                    "direction": direction
                }
            else:
                logger.info(f"{stage.capitalize()} model performance for metric '{metric}' is acceptable: expected <= {expected_value:.4f}, got {current_value:.4f}")
                performance_results[metric] = {
                    "status": "acceptable",
                    "expected": expected_value,
                    "current": current_value,
                    "direction": direction
                }
        else:
            msg = f"Invalid direction '{direction}' for metric '{metric}'. Must be 'higher' or 'lower'."
            logger.error(msg)
            raise ConfigError(msg)
    return performance_results

@dataclass
class MonitoringExecutionOutput:
    drift_results: dict[str, float]
    performance_results: dict[str | MetricName, dict[str, Any]]
    model_version: str
def prepare_metadata(
    *,
    args: argparse.Namespace,
    run_id: str,
    timestamp: datetime,
    prod_monitoring_output: MonitoringExecutionOutput | None = None,
    stage_monitoring_output: MonitoringExecutionOutput | None = None
) -> dict[str, Any]:
    metadata = {
        "problem_type": args.problem,
        "segment": args.segment,
        "timestamp": timestamp.isoformat(),
        "run_id": run_id,
        "inference_run_id": args.inference_run_id,
    }

    if prod_monitoring_output:
        metadata["production"] = prod_monitoring_output.__dict__

    if stage_monitoring_output:
        metadata["staging"] = stage_monitoring_output.__dict__

    return metadata

def execute_monitoring(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"],
    promotion_metrics_info: PromotionMetricsConfig
) -> MonitoringExecutionOutput:
    training_features = load_training_features(args=args, model_metadata=model_metadata)
    inference_features_and_target = load_inference_features_and_target(args, model_metadata, stage=stage)
    inference_features = inference_features_and_target.features
    target = inference_features_and_target.target

    drift_results = compare_feature_distributions(
        stage=stage,
        inference_features=inference_features,
        training_features=training_features
    )

    performance_results = assess_model_performance(
        args=args,
        model_metadata=model_metadata,
        stage=stage,
        target=target,
        promotion_metrics_info=promotion_metrics_info
    )

    output = MonitoringExecutionOutput(
        drift_results=drift_results,
        performance_results=performance_results,
        model_version=model_metadata.model_version
    )

    return output

def compare_production_and_staging_performance(
    prod_monitoring_output: MonitoringExecutionOutput,
    stage_monitoring_output: MonitoringExecutionOutput
) -> dict[str, Any]:
    performance_comparison: dict[str | MetricName, dict[str, Any]] = {}

    for metric in prod_monitoring_output.performance_results:
        if metric not in stage_monitoring_output.performance_results:
            logger.warning(f"Metric '{metric}' is present in production monitoring results but missing in staging monitoring results. Skipping comparison for this metric.")
            continue
        prod_metric_info = prod_monitoring_output.performance_results[metric]
        stage_metric_info = stage_monitoring_output.performance_results[metric]

        prod_perf = prod_metric_info["current"]
        stage_perf = stage_metric_info["current"]

        if not isinstance(prod_perf, (int, float)) or not isinstance(stage_perf, (int, float)):
            msg = f"Current performance values for metric '{metric}' must be numeric for comparison. Got production: {prod_perf} ({type(prod_perf)}), staging: {stage_perf} ({type(stage_perf)})."
            logger.error(msg)
            raise PipelineContractError(msg)

        if prod_metric_info["direction"] != stage_metric_info["direction"]:
            msg = f"Direction for metric '{metric}' is different between production and staging: '{prod_metric_info['direction']}' vs '{stage_metric_info['direction']}'. This should not happen."
            logger.error(msg)
            raise ConfigError(msg)

        direction = prod_metric_info["direction"]
        if direction == "maximize":
            diff = stage_perf - prod_perf
            status = "improvement" if diff > 0 else "degradation" if diff < 0 else "no_change"
        elif direction == "minimize":
            diff = prod_perf - stage_perf
            status = "improvement" if diff > 0 else "degradation" if diff < 0 else "no_change"
        else:
            msg = f"Invalid direction '{direction}' for metric '{metric}'. Must be 'maximize' or 'minimize'."
            logger.error(msg)
            raise ConfigError(msg)

        performance_comparison[metric] = {
            "production": prod_perf,
            "staging": stage_perf,
            "difference": diff,
            "status": status,
            "direction": direction
        }

    return performance_comparison


def main() -> int:
    args = parse_args()

    timestamp = datetime.now()
    run_id = f"{iso_no_colon(timestamp)}_{uuid4().hex[:8]}"
    base_path = Path("monitoring") / args.problem / args.segment

    run_dir = base_path / run_id

    log_path = run_dir / "monitor.log"

    setup_logging(log_path, getattr(logging, args.logging_level, logging.INFO))

    try:
        promotion_metrics_info = get_promotion_metrics_info(args)
        model_registry_info = get_model_registry_info(args)

        prod_meta = model_registry_info.prod_meta
        stage_meta = model_registry_info.stage_meta

        if prod_meta is None and stage_meta is None:
            msg = f"No production or staging model registry entry found for problem '{args.problem}' and segment '{args.segment}'. Cannot perform monitoring."
            logger.error(msg)
            raise PipelineContractError(msg)

        prod_monitoring_output, stage_monitoring_output = None, None

        if prod_meta:
            prod_monitoring_output = execute_monitoring(
                args=args,
                model_metadata=prod_meta,
                stage="production",
                promotion_metrics_info=promotion_metrics_info
            )
        if stage_meta:
            stage_monitoring_output = execute_monitoring(
                args=args,
                model_metadata=stage_meta,
                stage="staging",
                promotion_metrics_info=promotion_metrics_info
            )

        metadata = prepare_metadata(
            args=args,
            run_id=run_id,
            timestamp=timestamp,
            prod_monitoring_output=prod_monitoring_output,
            stage_monitoring_output=stage_monitoring_output
        )

        if prod_monitoring_output and stage_monitoring_output:
            performance_comparison = compare_production_and_staging_performance(prod_monitoring_output, stage_monitoring_output)
            metadata["staging_vs_production_comparison"] = performance_comparison

        save_metadata(metadata, target_dir=run_dir)

        return 0
    except Exception as e:
        logger.exception("Monitoring failed")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())
