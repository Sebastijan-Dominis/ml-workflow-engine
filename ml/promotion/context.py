import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.promotion.constants.constants import RunnersMetadata
from ml.utils.iso_no_col import iso_no_colon


@dataclass
class PromotionPaths:
    model_registry_dir: Path
    run_dir: Path
    promotion_configs_dir: Path
    train_run_dir: Path
    eval_run_dir: Path
    explain_run_dir: Path
    registry_path: Path
    archive_path: Path


@dataclass
class PromotionContext:
    args: argparse.Namespace
    run_id: str
    timestamp: str
    paths: PromotionPaths
    runners_metadata: RunnersMetadata | None = None


def build_context(args: argparse.Namespace) -> PromotionContext:
    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"

    model_registry_dir = Path("model_registry")

    paths = PromotionPaths(
        model_registry_dir=model_registry_dir,
        run_dir=model_registry_dir / "runs" / run_id,
        promotion_configs_dir=Path("configs") / "promotion",
        train_run_dir=Path("experiments") / args.problem / args.segment / args.version / args.experiment_id / "training" / args.train_run_id,
        eval_run_dir=Path("experiments") / args.problem / args.segment / args.version / args.experiment_id / "evaluation" / args.eval_run_id,
        explain_run_dir=Path("experiments") / args.problem / args.segment / args.version / args.experiment_id / "explainability" / args.explain_run_id,
        registry_path=model_registry_dir / "models.yaml",
        archive_path=model_registry_dir / "archive.yaml",
    )

    paths.run_dir.mkdir(parents=True, exist_ok=False)

    return PromotionContext(
        args=args,
        run_id=run_id,
        timestamp=timestamp,
        paths=paths,
        runners_metadata=None,
    )