"""A module for preparing metadata for post-promotion monitoring."""

import argparse
from datetime import datetime
from typing import Any

from ml.post_promotion.monitoring.classes.function_returns import MonitoringExecutionOutput


def prepare_metadata(
    *,
    args: argparse.Namespace,
    run_id: str,
    timestamp: datetime,
    prod_monitoring_output: MonitoringExecutionOutput | None = None,
    stage_monitoring_output: MonitoringExecutionOutput | None = None
) -> dict[str, Any]:
    """Prepare metadata for post-promotion monitoring.

    Args:
        args: Command-line arguments containing necessary identifiers.
        run_id: The ID of the monitoring run.
        timestamp: The timestamp of the monitoring run.
        prod_monitoring_output: The monitoring output from the production model.
        stage_monitoring_output: The monitoring output from the staging model.

    Returns:
        A dictionary containing the prepared metadata.
    """
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
