# """CLI for running production model inference with scalable batch storage.

# - Single-row or batch input.
# - Updates daily/hourly prediction batch files.
# - Designed so LLMs or other programs can call the core engine without storage or CLI logic.
# """

# import argparse
# import hashlib
# import logging
# import sys
# from datetime import datetime
# from pathlib import Path
# from typing import Any

# import numpy as np
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
# import yaml
# from ml.cli.error_handling import resolve_exit_code
# from ml.exceptions import PipelineContractError
# from ml.runners.shared.loading.pipeline import load_model_or_pipeline
# from ml.utils.hashing.service import hash_artifact

# logger = logging.getLogger(__name__)

# # -----------------------
# # Core Prediction Engine
# # -----------------------
# class InferenceEngine:
#     """Encapsulates model/pipeline loading, validation, and prediction logic."""

#     def __init__(self, artifact_meta: dict[str, Any]):
#         self.artifact_meta = artifact_meta
#         self.pipeline_path = Path(artifact_meta["artifacts"].get("pipeline_path", ""))
#         self.model_path = Path(artifact_meta["artifacts"].get("model_path", ""))
#         self.expected_hash = artifact_meta["artifacts"].get(
#             "pipeline_hash" if self.pipeline_path.exists() else "model_hash"
#         )
#         self._artifact: Any = None
#         self._load_and_validate_artifact()
#         self.feature_columns = [
#             f["name"] for f in artifact_meta.get("feature_lineage", [])
#         ]

#     def _load_and_validate_artifact(self) -> None:
#         """Load pipeline or model and validate hash."""
#         if self.pipeline_path.exists():
#             self._artifact = load_model_or_pipeline(self.pipeline_path, target_type="pipeline")
#             actual_hash = hash_artifact(self.pipeline_path)
#         elif self.model_path.exists():
#             self._artifact = load_model_or_pipeline(self.model_path, target_type="model")
#             actual_hash = hash_artifact(self.model_path)
#             pass  # non-pipeline model logic can be implemented here
#         else:
#             raise PipelineContractError("No valid model or pipeline artifact found.")

#         if actual_hash != self.expected_hash:
#             raise PipelineContractError(f"Artifact hash mismatch! Expected {self.expected_hash}, got {actual_hash}")
#         logger.info("Artifact loaded and validated successfully.")

#     def _normalize_input(self, X: Any) -> pd.DataFrame:
#         """Normalize input into a DataFrame with expected columns."""
#         if isinstance(X, pd.DataFrame):
#             df = X.copy()
#         elif isinstance(X, dict):
#             df = pd.DataFrame([X])
#         elif isinstance(X, list):
#             if all(isinstance(r, dict) for r in X):
#                 df = pd.DataFrame(X)
#             else:
#                 df = pd.DataFrame([X], columns=self.feature_columns)
#         else:
#             raise ValueError(f"Unsupported input type: {type(X)}")
#         return df[self.feature_columns]

#     def _hash_input_row(self, row: pd.Series) -> str:
#         """Return SHA-256 hash for a row to trace predictions."""
#         row_hash = pd.util.hash_pandas_object(row, index=True).values
#         row_bytes = np.asarray(row_hash).tobytes()
#         return hashlib.sha256(row_bytes).hexdigest()

#     # -----------------------
#     # Public Prediction APIs
#     # -----------------------
#     def predict(self, X: Any) -> pd.Series:
#         df = self._normalize_input(X)
#         return pd.Series(self._artifact.predict(df), index=df.index)

#     def predict_proba(self, X: Any) -> pd.DataFrame:
#         df = self._normalize_input(X)
#         return pd.DataFrame(self._artifact.predict_proba(df), index=df.index)

#     # -----------------------
#     # Batch Storage (CLI only)
#     # -----------------------
#     def store_predictions(
#         self,
#         X: pd.DataFrame,
#         predictions: pd.Series,
#         probabilities: pd.DataFrame,
#         output_dir: Path,
#         time_window: str = "daily",
#     ) -> Path:
#         """
#         Store predictions in daily/hourly Parquet batch file. Updates existing file if present.

#         Args:
#             X: Input DataFrame
#             predictions: pd.Series of predictions
#             probabilities: pd.DataFrame of probabilities
#             output_dir: Directory to store batch files
#             time_window: "hourly" or "daily"

#         Returns:
#             Path to updated Parquet batch file
#         """
#         output_dir.mkdir(parents=True, exist_ok=True)
#         now = datetime.utcnow()
#         if time_window == "hourly":
#             batch_file = output_dir / now.strftime("batch_%Y-%m-%d_%H.parquet")
#         elif time_window == "daily":
#             batch_file = output_dir / now.strftime("batch_%Y-%m-%d.parquet")
#         else:
#             raise ValueError("time_window must be 'hourly' or 'daily'")

#         # Build new records
#         new_df = X.copy()
#         new_df["prediction"] = predictions.values
#         new_df["prediction_proba"] = probabilities.apply(lambda row: row.tolist(), axis=1)
#         new_df["input_hash"] = X.apply(self._hash_input_row, axis=1)
#         new_df["timestamp"] = now.isoformat()

#         # Append to existing Parquet file if exists
#         if batch_file.exists():
#             existing_table = pq.read_table(batch_file)
#             existing_df = existing_table.to_pandas()
#             combined_df = pd.concat([existing_df, new_df], ignore_index=True)
#             pq.write_table(pa.Table.from_pandas(combined_df), batch_file)
#         else:
#             pq.write_table(pa.Table.from_pandas(new_df), batch_file)

#         logger.info(f"Stored/updated Parquet batch predictions at {batch_file}")
#         return batch_file


# # -----------------------
# # CLI Entry Point
# # -----------------------
# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Run production inference and store batch predictions.")
#     parser.add_argument("--problem_type", type=str, required=True)
#     parser.add_argument("--segment", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, default="predictions")
#     parser.add_argument("--logging-level", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], default="INFO")
#     parser.add_argument("--features", type=str, nargs="*", help="Single-row features as key=value")
#     parser.add_argument("--time_window", type=str, choices=["hourly","daily"], default="daily")
#     return parser.parse_args()


# def main() -> int:
#     args = parse_args()
#     logging.basicConfig(level=getattr(logging, args.logging_level.upper(), logging.INFO),
#                         format="%(asctime)s [%(levelname)s] %(message)s")

#     try:
#         # Load artifact metadata
#         registry_path = Path("model_registry.yaml")
#         if not registry_path.exists():
#             raise PipelineContractError(f"Model registry not found at {registry_path}")
#         with open(registry_path) as f:
#             registry = yaml.safe_load(f)
#         prod_meta = registry[args.problem_type][args.segment]["production"]

#         engine = InferenceEngine(prod_meta)



#         # Predictions
#         preds = engine.predict(X_input)
#         probs = engine.predict_proba(X_input)

#         # Store predictions (CLI only)
#         engine.store_predictions(X_input, preds, probs, Path(args.output_dir), time_window=args.time_window)

#         return 0  # EXIT_SUCCESS

#     except Exception as e:
#         logger.exception("Inference failed")
#         return resolve_exit_code(e)


# if __name__ == "__main__":
#     sys.exit(main())
