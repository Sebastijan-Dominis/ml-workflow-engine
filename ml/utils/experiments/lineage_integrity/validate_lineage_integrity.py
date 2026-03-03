"""Orchestration entrypoint for lineage integrity validation checks."""

from pathlib import Path

from ml.config.schemas.model_cfg import (SearchModelConfig,
                                                    TrainModelConfig)
from ml.utils.experiments.lineage_integrity.validations.base import \
    validate_base_lineage_integrity
from ml.utils.experiments.lineage_integrity.validations.configs_match import \
    validate_configs_match


def validate_lineage_integrity(source_dir: Path, cfg: TrainModelConfig | SearchModelConfig | None = None) -> None:
    """Run baseline lineage checks and optional config-hash consistency validation.

    Args:
        source_dir: Directory containing run metadata to validate.
        cfg: Optional validated config for hash consistency validation.

    Returns:
        None.
    """

    validate_base_lineage_integrity(source_dir)

    if cfg:
        validate_configs_match(source_dir, cfg)