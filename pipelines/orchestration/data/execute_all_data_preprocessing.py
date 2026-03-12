"""Batch runner for end-to-end data preprocessing.

This script orchestrates three pipeline stages across discovered datasets and
configurations: raw metadata handling, interim dataset generation, and
processed dataset generation.
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.formatting.str_to_bool import str_to_bool
from ml.logging_config import setup_logging

from pipelines.orchestration.common.orchestration_logging import log_completion

logger = logging.getLogger(__name__)


def run_cmd(cmd: list[str]) -> None:
    """Execute a subprocess command and log captured standard output.

    Args:
        cmd: Command and arguments to execute.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero
            status code.
    """
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the preprocessing orchestrator.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run the full data preprocessing pipeline, including handling raw data, making interim data, and making processed data. This script will look for all raw data snapshots in the data/raw directory, all interim config files in configs/data/interim, and all processed config files in configs/data/processed, and run the corresponding scripts for each of them.")

    parser.add_argument(
        "--skip-if-existing",
        type=str_to_bool,
        default=True,
        help="Whether to skip running a data preprocessing step if the expected output already exists (e.g., metadata.json for register_raw_snapshot) (default: True)"
    )

    return parser.parse_args()

def main() -> int:
    """Run data preprocessing steps for all discoverable datasets.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        The workflow is best-effort per stage ordering (raw -> interim ->
        processed) and supports idempotent skipping when prior outputs exist.

    Side Effects:
        Executes multiple subprocess pipelines, writes consolidated script logs,
        and may create new snapshot artifacts across data stages.

    Examples:
        python -m pipelines.orchestration.data.execute_all_data_preprocessing --skip-if-existing true
    """
    args = parse_args()

    start_time = time.perf_counter()

    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"

    log_path = Path(f"orchestration_logs/data/execute_all_data_preprocessing/{run_id}/data_preprocessing.log")
    setup_logging(log_path)

    logger.info(f"Starting the full data preprocessing run: {run_id}")

    # register_raw_snapshot -> build_interim_dataset -> build_processed_dataset
    try:
        raw_root = Path("data/raw")

        if raw_root.exists():
            for data_dir in sorted(raw_root.iterdir()):
                if not data_dir.is_dir():
                    continue

                data_name = data_dir.name

                for version_dir in sorted(data_dir.iterdir()):
                    if not version_dir.is_dir():
                        continue

                    data_version = version_dir.name

                    for snapshot_dir in sorted(version_dir.iterdir()):
                        if not snapshot_dir.is_dir():
                            continue

                        snapshot_id = snapshot_dir.name

                        metadata_path = snapshot_dir / "metadata.json"

                        if metadata_path.exists() and args.skip_if_existing:
                            logger.info(
                                f"Skipping register_raw_snapshot for {data_name} v{data_version} "
                                f"snapshot {snapshot_id} (metadata exists)"
                            )
                            continue

                        cmd = [
                            sys.executable,
                            "-m",
                            "pipelines.data.register_raw_snapshot",
                            "--data", data_name,
                            "--version", data_version,
                            "--snapshot_id", snapshot_id,
                        ]

                        logger.info(f"Starting register_raw_snapshot for {data_name} v{data_version} snapshot {snapshot_id}")
                        run_cmd(cmd)
                        logger.info(f"Completed register_raw_snapshot for {data_name} v{data_version} snapshot {snapshot_id}")

        interim_config_root = Path("configs/data/interim")

        if interim_config_root.exists():
            for data_dir in sorted(interim_config_root.iterdir()):
                if not data_dir.is_dir():
                    continue

                data_name = data_dir.name

                for config_file in sorted(data_dir.glob("*.yaml")):
                    version = config_file.stem

                    interim_output_root = Path("data/interim") / data_name / version

                    existing_runs = (
                        [d for d in interim_output_root.iterdir() if d.is_dir()]
                        if interim_output_root.exists()
                        else []
                    )

                    if existing_runs and args.skip_if_existing:
                        logger.info(
                            f"Skipping build_interim_dataset for {data_name} v{version} "
                            f"(existing runs: {[d.name for d in existing_runs]})"
                        )
                        continue

                    cmd = [
                        sys.executable,
                        "-m",
                        "pipelines.data.build_interim_dataset",
                        "--data", data_name,
                        "--version", version,
                    ]

                    logger.info(f"Starting build_interim_dataset for {data_name} v{version}")
                    run_cmd(cmd)
                    logger.info(f"Completed build_interim_dataset for {data_name} v{version}")

        processed_config_root = Path("configs/data/processed")

        if processed_config_root.exists():
            for data_dir in sorted(processed_config_root.iterdir()):
                if not data_dir.is_dir():
                    continue

                data_name = data_dir.name

                for config_file in sorted(data_dir.glob("*.yaml")):
                    version = config_file.stem

                    processed_output_root = Path("data/processed") / data_name / version

                    existing_runs = (
                        [d for d in processed_output_root.iterdir() if d.is_dir()]
                        if processed_output_root.exists()
                        else []
                    )

                    if existing_runs and args.skip_if_existing:
                        logger.info(
                            f"Skipping build_processed_dataset for {data_name} v{version} "
                            f"(existing runs: {[d.name for d in existing_runs]})"
                        )
                        continue

                    cmd = [
                        sys.executable,
                        "-m",
                        "pipelines.data.build_processed_dataset",
                        "--data", data_name,
                        "--version", version,
                    ]

                    logger.info(f"Starting build_processed_dataset for {data_name} v{version}")
                    run_cmd(cmd)
                    logger.info(f"Completed build_processed_dataset for {data_name} v{version}")

        log_completion(start_time=start_time, message="Full data preprocessing run completed successfully.")
        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while running a subprocess command: {e}")
        msg = f"Command '{' '.join(e.cmd)}' failed with exit code {e.returncode}"
        log_completion(start_time=start_time, message=msg)
        return e.returncode

    except Exception:
        logger.exception("Unexpected error during the data preprocessing run.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
