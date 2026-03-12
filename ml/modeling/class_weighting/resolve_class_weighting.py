"""Resolve class-weight parameters from policy, data statistics, and library."""

import logging

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import ConfigError
from ml.modeling.class_weighting.constants import SUPPORTED_LIBRARIES
from ml.modeling.class_weighting.models import DataStats


def resolve_class_weighting(
        config: TrainModelConfig | SearchModelConfig,
        stats: DataStats,
        library: SUPPORTED_LIBRARIES
    ) -> dict:
    """Return library-specific class-weight parameters based on configured policy.

    Args:
        config: Validated training or search configuration with class-weighting settings.
        stats: Dataset class distribution statistics.
        library: Target ML library receiving class-weight parameters.

    Returns:
        Dictionary of class-weight parameters for the selected library.
    """

    policy = config.class_weighting.policy

    if policy == "off":
        logging.info("Class weighting is turned off.")
        return {}

    if policy == "if_imbalanced":
        if config.class_weighting.imbalance_threshold is None:
            msg = "imbalance_threshold must be set when using 'if_imbalanced' policy."
            logging.error(msg)
            raise ConfigError(msg)
        if stats.minority_ratio >= config.class_weighting.imbalance_threshold:
            logging.info(f"Minority ratio {stats.minority_ratio:.4f} is above imbalance threshold {config.class_weighting.imbalance_threshold:.4f}. Class weighting will not be applied.")
            return {}

    if config.class_weighting.strategy == "ratio":
        neg, pos = stats.class_counts[0], stats.class_counts[1]
        ratio = neg / pos

        if library in ["xgboost", "lightgbm"]:
            logging.info(f"Using ratio class weighting strategy for {library}. Ratio: {ratio}")
            return {"scale_pos_weight": ratio}
        if library == "catboost":
            logging.info(f"Using ratio class weighting strategy for {library}. Ratio: {ratio}")
            return {"class_weights": [1.0, ratio]}

    if config.class_weighting.strategy == "balanced":
        classes = np.array(list(stats.class_counts.keys()))
        y_expanded = np.concatenate([
            np.full(v, k) for k, v in stats.class_counts.items()
        ])
        weights = compute_class_weight("balanced", classes=classes, y=y_expanded)
        logging.info("Using balanced class weighting strategy. Computed class weights: %s", dict(zip(classes, weights, strict=False)))
        return {"class_weights": weights.tolist()}

    msg = f"Unsupported class weighting strategy: {config.class_weighting.strategy}"
    logging.error(msg)
    raise ConfigError(msg)
