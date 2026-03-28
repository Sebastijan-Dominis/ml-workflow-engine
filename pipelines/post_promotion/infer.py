"""CLI for running model inference (production + staging) with monitoring-ready outputs."""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.cli.error_handling import resolve_exit_code
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.logging_config import setup_logging
from ml.post_promotion.inference.execution.execute_inference import execute_inference
from ml.post_promotion.shared.loading.model_registry import get_model_registry_info

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
        "--snapshot-bindings-id",
        required=True,
        help = "A snapshot binding to define which snapshot to load for each feature set."
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
    base_path = Path("predictions") / args.problem / args.segment

    run_dir = base_path / run_id

    log_path = run_dir / "inference.log"

    setup_logging(log_path, getattr(logging, args.logging_level, logging.INFO))

    try:
        model_registry_info = get_model_registry_info(args)

        prod_meta = model_registry_info.prod_meta
        stage_meta = model_registry_info.stage_meta

        if prod_meta is not None:
            execute_inference(
                args=args,
                model_metadata=prod_meta,
                stage="production",
                timestamp=timestamp,
                path=run_dir / "production",
                run_id=run_id
            )

        if stage_meta is not None:
            execute_inference(
                args=args,
                model_metadata=stage_meta,
                stage="staging",
                timestamp=timestamp,
                path=run_dir / "staging",
                run_id=run_id
            )

        return 0

    except Exception as e:
        logger.exception("Inference failed")
        return resolve_exit_code(e)


if __name__ == "__main__":
    sys.exit(main())
