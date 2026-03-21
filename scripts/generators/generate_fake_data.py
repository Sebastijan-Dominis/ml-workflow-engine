"""Utility CLI to generate synthetic tabular data based on an existing dataset snapshot.

This script loads a dataset snapshot, trains a statistical synthesizer model,
and generates synthetic rows that resemble the original data distribution.

Enhancements:
    - Temporal continuity (supports datetime + decomposed date columns)
    - Schema enforcement (dtype + categorical alignment)
    - Constraint enforcement for logical consistency
    - Outlier clipping for stability

Examples:
    python -m scripts.generators.generate_fake_data \
        --data hotel_bookings \
        --version v1 \
        --num_rows 5000 \
        --include_old true
"""

import argparse
import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.formatting.str_to_bool import str_to_bool
from ml.logging_config import setup_logging
from ml.types import LatestSnapshot
from ml.utils.loaders import read_data
from ml.utils.snapshots.snapshot_path import get_snapshot_path
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

logger = logging.getLogger(__name__)


MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

REVERSE_MONTH_MAP = {v: k for k, v in MONTH_MAP.items()}


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic dataset snapshot.")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Name of the dataset to use (e.g., 'hotel_bookings')."
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version of the dataset to use (e.g., 'v1')."
    )

    parser.add_argument(
        "--snapshot-id",
        type=str,
        default=LatestSnapshot.LATEST.value,
        help="Snapshot ID to use as base for synthetic generation. Defaults to 'latest'.",
    )

    parser.add_argument(
        "--num-rows",
        type=int,
        default=500,
        help="Number of synthetic rows to generate. Defaults to 500.",
    )

    parser.add_argument(
        "--include-old",
        type=str_to_bool,
        default=False,
        help="If true, final dataset will include original rows plus synthetic. If false, only synthetic rows are saved.",
    )

    parser.add_argument(
        "--strict-missing",
        type=str_to_bool,
        default=False,
        help="If true, abort if final dataset contains any missing values after processing. If false, log a warning and proceed with saving.",
    )

    parser.add_argument(
        "--strict-quality",
        type=str_to_bool,
        default=True,
        help="If true, abort if the synthetic data quality score is below the specified threshold. If false, log a warning and proceed with saving.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Defaults to 42.",
    )

    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.7,
        help="Minimum acceptable quality score for synthetic data. If the generated data scores below this threshold, the script will either log a warning or abort, depending on the '--strict-quality' flag.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Number of training epochs for the synthesizer model. Defaults to 400.",
    )

    parser.add_argument(
        "--batch-target-size",
        type=int,
        default=4000,
        help="Target batch size for synthesizer sampling. If the synthesizer does not support direct batch size control, this will be used to generate samples in batches and concatenate them. Defaults to 4000.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a pre-trained CTGAN model to load. If not provided, train a new model."
    )

    parser.add_argument(
        "--save-model",
        type=str_to_bool,
        default=True,
        help="Whether to save the trained CTGAN model to disk if training a new one."
    )

    return parser.parse_args()


def _infer_format(file_path: Path) -> str:
    """Infer file format from extension."""
    suffix = file_path.suffix.lower().replace(".", "")
    if not suffix:
        raise ValueError(f"Could not infer format from {file_path}")
    return suffix


def _atomic_save(df: pd.DataFrame, path: Path, fmt: str) -> None:
    """Atomically save dataframe."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        if fmt == "csv":
            df.to_csv(tmp_path, index=False)
        elif fmt == "parquet":
            df.to_parquet(tmp_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        tmp_path.replace(path)

    except Exception as e:
        logger.exception(f"Atomic save failed for {path}")
        raise RuntimeError(f"Failed saving {path}") from e


# -------------------- TEMPORAL LOGIC -------------------- #

def _handle_temporal_flow(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> pd.DataFrame:
    """Ensure synthetic data continues forward in time with robust NA handling."""

    # ------------------ CASE 1: TRUE DATETIME ------------------ #
    datetime_cols = df_real.select_dtypes(include=["datetime64[ns]"]).columns

    if len(datetime_cols) > 0:
        for col in datetime_cols:
            max_real = df_real[col].max()
            min_synth = df_synth[col].min()

            if pd.notna(max_real) and pd.notna(min_synth):
                delta = max_real - min_synth + pd.Timedelta(days=1)
                df_synth[col] = df_synth[col] + delta

            # Log NaT issues
            n_missing = df_synth[col].isna().sum()
            if n_missing > 0:
                logger.warning(f"{col}: {n_missing} NaT values after temporal shift")

        logger.info("Applied datetime-based temporal shift")
        return df_synth

    # ------------------ CASE 2: DECOMPOSED DATES ------------------ #
    required = {
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month"
    }

    if required.issubset(df_real.columns):

        def build_datetime(df: pd.DataFrame):
            months = df["arrival_date_month"].map(MONTH_MAP)

            date_df = pd.DataFrame({
                "year": pd.to_numeric(df["arrival_date_year"], errors="coerce"),
                "month": months,
                "day": pd.to_numeric(df["arrival_date_day_of_month"], errors="coerce"),
            })

            return pd.to_datetime(date_df, errors="coerce")

        real_dates = build_datetime(df_real)
        synth_dates = build_datetime(df_synth)

        max_real = real_dates.max()
        min_synth = synth_dates.min()

        if pd.notna(max_real) and pd.notna(min_synth):
            delta = max_real - min_synth + pd.Timedelta(days=1)
            shifted = synth_dates + delta
        else:
            logger.warning("Temporal shift skipped due to invalid min/max dates")
            shifted = synth_dates

        # Log invalid dates
        n_invalid = shifted.isna().sum()
        if n_invalid > 0:
            logger.warning(f"{n_invalid} invalid synthetic dates detected")

        # Fill invalid dates forward (safe fallback)
        shifted = shifted.ffill().bfill()

        # Rebuild columns
        df_synth["arrival_date_year"] = shifted.dt.year.astype("Int64")
        df_synth["arrival_date_month"] = shifted.dt.month.map(REVERSE_MONTH_MAP)
        df_synth["arrival_date_day_of_month"] = shifted.dt.day.astype("Int64")

        # ✅ SAFE WEEK NUMBER
        df_synth["arrival_date_week_number"] = (
            shifted.dt.isocalendar().week.astype("Int64")
        )

        logger.info("Applied decomposed-date temporal shift")
        return df_synth

    logger.info("No temporal structure detected")
    return df_synth


# -------------------- DATA QUALITY -------------------- #

def _clip_outliers(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> pd.DataFrame:
    """Clip synthetic values to realistic ranges with logging."""
    numeric_cols = df_real.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        q_low = df_real[col].quantile(0.01)
        q_high = df_real[col].quantile(0.99)

        before_outliers = ((df_synth[col] < q_low) | (df_synth[col] > q_high)).sum()

        df_synth[col] = df_synth[col].clip(q_low, q_high)

        if before_outliers > 0:
            logger.info(f"{col}: clipped {before_outliers} outliers")

    return df_synth


def _apply_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain constraints."""
    if "adults" in df.columns:
        df["adults"] = df["adults"].clip(lower=1)

    if "children" in df.columns:
        df["children"] = df["children"].clip(lower=0)

    if "adr" in df.columns:
        df["adr"] = df["adr"].clip(lower=0)

    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(df.columns):
        total = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        df.loc[total == 0, "stays_in_week_nights"] = 1

    return df


def _enforce_schema(df: pd.DataFrame, dtypes: dict, categories: dict) -> pd.DataFrame:
    """Reapply original schema with robust coercion + logging."""

    for col, dtype in dtypes.items():
        if col not in df.columns:
            continue

        try:
            if "int" in str(dtype):
                coerced = pd.to_numeric(df[col], errors="coerce")
                n_na = coerced.isna().sum()

                if n_na > 0:
                    logger.warning(f"{col}: {n_na} values could not be cast to int")

                df[col] = coerced.astype("Int64")

            elif "float" in str(dtype):
                coerced = pd.to_numeric(df[col], errors="coerce")
                df[col] = coerced.astype(dtype)

            else:
                df[col] = df[col].astype(dtype)

        except Exception:
            logger.warning(f"Failed to cast column {col} to {dtype}")

    # ------------------ CATEGORICAL ALIGNMENT ------------------ #
    for col, cats in categories.items():
        if col not in df.columns:
            continue

        mask_invalid = ~df[col].isin(cats)
        n_invalid = mask_invalid.sum()

        if n_invalid > 0:
            logger.warning(f"{col}: {n_invalid} unseen categories replaced with NaN")

        df.loc[mask_invalid, col] = pd.NA

    return df

def _preprocess_for_sdv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numerical → median
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical → "missing"
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("missing")

    return df

# -------------------- MAIN -------------------- #

def main() -> int:
    """Main execution."""
    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    log_file = Path(f"scripts_logs/generators/generate_fake_data/{run_id}/generation.log")
    setup_logging(path=log_file, level=logging.INFO)

    args = parse_args()

    logger.info(f"Using random seed: {args.seed}")
    np.random.seed(args.seed)
    random.seed(args.seed)

    try:
        base_path = Path(f"data/raw/{args.data}/{args.version}")
        snapshot_path = get_snapshot_path(args.snapshot_id, base_path)

        logger.info(f"Resolved snapshot path: {snapshot_path}")

        data_files = list(snapshot_path.glob("data.*"))
        if not data_files:
            msg = f"No data file found in {snapshot_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        data_file = data_files[0]
        fmt = _infer_format(data_file)

        df = read_data(fmt, data_file)
        if df is None:
            msg = f"read_data returned None for file {data_file}"
            logger.error(msg)
            raise RuntimeError(msg)
        logger.info(f"Loaded dataset shape: {df.shape}")

        # Capture schema
        original_dtypes = df.dtypes.to_dict()
        original_categories = {
            col: df[col].dropna().unique()
            for col in df.select_dtypes(include="object").columns
        }

        df_raw = df.copy()
        df_model = _preprocess_for_sdv(df_raw)

        DROP_COLS = ["name", "email", "phone-number", "credit_card"]

        df_model = df_model.drop(columns=[c for c in DROP_COLS if c in df_model.columns])

        # Simplify 'country': keep top 20, others as 'Other'
        if "country" in df_model.columns:
            top_countries = df_model["country"].value_counts().nlargest(20).index
            df_model["country"] = df_model["country"].where(df_model["country"].isin(top_countries), "Other")

        # Convert 'reservation_status_date' to month-year period to reduce cardinality
        if "reservation_status_date" in df_model.columns:
            df_model["reservation_status_date"] = pd.to_datetime(df_model["reservation_status_date"], errors="coerce")
            df_model["reservation_status_month"] = df_model["reservation_status_date"].dt.to_period("M").astype(str)
            df_model.drop(columns=["reservation_status_date"], inplace=True)

        # Fill sparse numeric columns with 0
        for col in ["agent", "company"]:
            if col in df_model.columns:
                df_model[col] = df_model[col].fillna(0)

        # Ensure no NaNs remain before training
        df_model.fillna("Unknown", inplace=True)

        new_snapshot_id = run_id
        output_dir = base_path / new_snapshot_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train synthesizer
        metadata = Metadata()

        metadata.detect_table_from_dataframe(
            data=df_model,
            table_name="synthetic_data",
            infer_keys=None # type: ignore -> None is actually a valid value for infer_keys
        )

        metadata.set_primary_key(
            table_name="synthetic_data",
            column_name=None
        )

        for col in ["children", "babies"]:
            if col in df_model.columns:
                metadata.update_column(col, sdtype="numerical")

        for col in ["is_canceled", "is_repeated_guest"]:
            if col in df_model.columns:
                df_model[col] = df_model[col].astype("int")
                metadata.update_column(col, sdtype="numerical")

        metadata_path = output_dir / "synthesizer_metadata.json"
        metadata.save_to_json(metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")

        if args.model_path is not None:
            logger.info(f"Loading pre-trained model from {args.model_path}")
            try:
                with open(args.model_path, "rb") as f:
                    synthesizer = pickle.load(f)
            except Exception as e:
                logger.exception(f"Failed to load model from {args.model_path}")
                raise RuntimeError("Failed to load CTGAN model") from e

        else:
            logger.info("No pre-trained model provided. Training new CTGAN synthesizer.")
            use_gpu = torch.cuda.is_available()
            try:
                if use_gpu:
                    torch.zeros(1, device="cuda")  # test allocation
            except RuntimeError:
                use_gpu = False
            device_str = "GPU" if use_gpu else "CPU"
            logger.info(f"Training CTGAN on {device_str}")

            pac = 10  # default PacGAN value
            target = args.batch_target_size

            # Floor the target to the nearest multiple of pac
            batch_size = (target // pac) * pac
            batch_size = max(batch_size, pac)  # ensure batch size at least pac

            synthesizer = CTGANSynthesizer(
                metadata=metadata,
                epochs=args.epochs,
                verbose=True,
                enable_gpu=use_gpu,
                batch_size=batch_size
            )

            logger.info(f"Starting synthesizer training for {args.epochs} epochs with batch size {batch_size} and device {device_str}")

            try:
                synthesizer.fit(df_model)
            except Exception as e:
                logger.exception("Failed during synthesizer.fit()")
                raise RuntimeError("Model training failed") from e

                # Save trained model if desired
            if args.save_model:
                model_file = Path("synthetizers") / new_snapshot_id / "ctgan_model.pkl"
                model_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(model_file, "wb") as f:
                        pickle.dump(synthesizer, f)
                    logger.info(f"Saved trained CTGAN model to {model_file}")
                except Exception as e:
                    logger.warning(f"Failed to save CTGAN model: {e}")

        try:
            n_rows_per_sample = min(1000, args.num_rows)
            n_samples = -(-args.num_rows // n_rows_per_sample)  # ceil division
            synthetic_df = pd.concat(
                [synthesizer.sample(n_rows_per_sample) for _ in range(n_samples)],
                ignore_index=True
            ).iloc[:args.num_rows]  # trim to exact number of rows
        except Exception as e:
            logger.exception("Failed during synthesizer.sample()")
            raise RuntimeError("Failed to generate synthetic data") from e

        synthetic_df = _preprocess_for_sdv(synthetic_df)

        quality_report = evaluate_quality(
            real_data=df_model,
            synthetic_data=synthetic_df,
            metadata=metadata
        )

        score = quality_report.get_score()
        logger.info(f"Synthetic data quality score: {score:.4f}")

        if synthetic_df.isna().sum().sum() > 0:
            logger.warning(
                f"Synthetic data contains {synthetic_df.isna().sum().sum()} missing values"
            )

        if score < args.quality_threshold:
            msg = f"Synthetic data quality score {score:.4f} is below threshold {args.quality_threshold:.4f}"
            if args.strict_quality:
                logger.error(msg + ". Aborting.")
                raise ValueError(msg)
            else:
                logger.warning(msg + ". Proceeding with save (strict=False).")

        # Apply improvements
        synthetic_df = _handle_temporal_flow(df_model, synthetic_df)
        synthetic_df = _clip_outliers(df_model, synthetic_df)
        synthetic_df = _apply_constraints(synthetic_df)
        synthetic_df = _enforce_schema(synthetic_df, original_dtypes, original_categories)

        # Merge
        final_df = pd.concat([df_model, synthetic_df], ignore_index=True) if args.include_old else synthetic_df

        if final_df.isna().sum().sum() > 0:
            if args.strict_missing:
                msg = f"Final dataset contains {final_df.isna().sum().sum()} missing values in strict mode. Aborting."
                logger.error(msg)
                raise ValueError(msg)
            logger.warning("Final dataset still contains missing values. Strict mode is disabled, so proceeding with save.")

        # Save
        output_file = output_dir / f"data.{fmt}"
        _atomic_save(final_df, output_file, fmt)

        logger.info(f"Saved synthetic dataset to {output_file}")
        print(f"New snapshot created: {new_snapshot_id}")

        return 0

    except Exception:
        logger.exception("Failed to generate synthetic data.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
