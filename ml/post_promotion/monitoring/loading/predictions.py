"""Module for loading predictions for post-promotion monitoring."""
import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

from ml.utils.snapshots.snapshot_path import get_snapshot_path


def load_predictions(
    args: argparse.Namespace,
    stage: Literal["production", "staging"]
) -> pd.DataFrame:
    """Load predictions for post-promotion monitoring.

    Args:
        args: Command-line arguments containing necessary identifiers.
        stage: The stage for which to load predictions ('production' or 'staging').

    Returns:
        A DataFrame containing the predictions."""
    inference_dir = Path("predictions") / args.problem / args.segment
    snapshot_path = get_snapshot_path(args.inference_run_id, inference_dir)
    inference_run_dir = snapshot_path / stage
    inference_predictions_path = inference_run_dir / "predictions.parquet"
    predictions = pd.read_parquet(inference_predictions_path)
    return predictions
