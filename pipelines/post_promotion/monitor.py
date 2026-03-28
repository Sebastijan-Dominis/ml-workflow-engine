"""Main module for the post-promotion monitoring pipeline."""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.cli.error_handling import resolve_exit_code
from ml.exceptions import PipelineContractError
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.persistence.save_metadata import save_metadata
from ml.logging_config import setup_logging
from ml.post_promotion.monitoring.execution.execute_monitoring import execute_monitoring
from ml.post_promotion.monitoring.loading.promotion_metrics_info import get_promotion_metrics_info
from ml.post_promotion.monitoring.performance.comparison import (
    compare_production_and_staging_performance,
)
from ml.post_promotion.monitoring.persistence.prepare_metadata import prepare_metadata
from ml.post_promotion.shared.loading.model_registry import get_model_registry_info
from ml.types.latest import LatestSnapshot

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

def main() -> int:
    """Main function to execute the post-promotion monitoring pipeline.

    Returns:
        An integer exit code (0 for success, non-zero for failure).
    """
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
