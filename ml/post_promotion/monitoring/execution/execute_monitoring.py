"""Module for executing the post-promotion monitoring pipeline."""
import argparse
from typing import Literal

from ml.post_promotion.monitoring.classes.function_returns import MonitoringExecutionOutput
from ml.post_promotion.monitoring.feature_drifting.comparison import compare_feature_distributions
from ml.post_promotion.monitoring.loading.inference_features_and_target import (
    load_inference_features_and_target,
)
from ml.post_promotion.monitoring.loading.training_features import load_training_features
from ml.post_promotion.monitoring.performance.assessment import assess_model_performance
from ml.promotion.config.promotion_thresholds import PromotionMetricsConfig
from ml.promotion.config.registry_entry import RegistryEntry


def execute_monitoring(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"],
    promotion_metrics_info: PromotionMetricsConfig
) -> MonitoringExecutionOutput:
    """Execute the post-promotion monitoring pipeline for a given model and stage.

    Args:
        args: Command-line arguments containing necessary identifiers.
        model_metadata: Metadata for the model being evaluated.
        stage: The stage for which to execute monitoring ("production" or "staging").
        promotion_metrics_info: Configuration for promotion metrics.

    Returns:
        A MonitoringExecutionOutput object containing the results of the monitoring execution.
    """
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
