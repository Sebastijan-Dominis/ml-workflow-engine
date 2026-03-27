"""Module for loading training features for post-promotion monitoring."""
import argparse

import pandas as pd

from ml.post_promotion.shared.loading.features import prepare_features
from ml.promotion.config.registry_entry import RegistryEntry


def load_training_features(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
) -> pd.DataFrame:
    """Load training features for post-promotion monitoring.

    Args:
        args: Command-line arguments containing necessary identifiers.
        model_metadata: Metadata for the model being evaluated.

    Returns:
        A DataFrame containing the training features.
    """
    training_features_return = prepare_features(
        args=args,
        model_metadata=model_metadata
    )
    training_features = training_features_return.features
    return training_features
