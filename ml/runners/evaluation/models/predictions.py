"""Module defining Pydantic models for representing prediction artifacts and their associated metadata in the evaluation runner."""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class PredictionArtifacts(BaseModel):
    """Model representing the prediction artifacts for each data split."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

class PredictionsPaths(BaseModel):
    """Model representing the file paths to the persisted prediction artifacts."""
    train_predictions_path: str = Field(..., description="File path to the training predictions parquet file.")
    val_predictions_path: str = Field(..., description="File path to the validation predictions parquet file.")
    test_predictions_path: str = Field(..., description="File path to the test predictions parquet file.")

class PredictionsPathsAndHashes(PredictionsPaths):
    """Model representing the file paths and hashes of the persisted prediction artifacts."""
    train_predictions_hash: str = Field(..., description="Hash of the training predictions artifact for integrity verification.")
    val_predictions_hash: str = Field(..., description="Hash of the validation predictions artifact for integrity verification.")
    test_predictions_hash: str = Field(..., description="Hash of the test predictions artifact for integrity verification.")
