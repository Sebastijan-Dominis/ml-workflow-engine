import argparse
import logging
import sys

from ml.cli.error_handling import resolve_exit_code
from ml.logging_config import setup_logging
from ml.promotion.context import PromotionContext, build_context
from ml.promotion.service import PromotionService

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage or promote a model.")

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
        "--version",
        type=str,
        required=True,
        help="Model version, e.g., 'v1'"
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment id (directory name under experiments/{problem}/{segment}/{version})"
    )

    parser.add_argument(
        "--train-run-id",
        type=str,
        required=True,
        help="Train run id (directory name under experiments/{problem}/{segment}/{version}/{experiment_id}/training)"
    )

    parser.add_argument(
        "--eval-run-id",
        type=str,
        required=True,
        help="Eval run id (directory name under experiments/{problem}/{segment}/{version}/{experiment_id}/evaluation)"
    )

    parser.add_argument(
        "--explain-run-id",
        type=str,
        required=True,
        help="Explain run id (directory name under experiments/{problem}/{segment}/{version}/{experiment_id}/explainability)"
    )

    parser.add_argument(
        "--stage",
        choices=["staging", "production"],
        required=True,
        help="Stage of the promotion (staging or production)"
    )

    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    return parser.parse_args()

def main() -> int:
    args: argparse.Namespace
    context: PromotionContext
    service: PromotionService

    args = parse_args()

    context = build_context(args)

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    log_file = context.paths.run_dir / "promotion.log"
    setup_logging(log_file, log_level)

    try:
        service = PromotionService()
        service.run(context)
        return 0
    except Exception as e:
        logger.exception("An error occurred during promotion.")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())