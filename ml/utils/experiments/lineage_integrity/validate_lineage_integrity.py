from pathlib import Path

from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.utils.experiments.lineage_integrity.validations.base import validate_base_lineage_integrity
from ml.utils.experiments.lineage_integrity.validations.configs_match import validate_configs_match


def validate_lineage_integrity(source_dir: Path, cfg: TrainModelConfig | SearchModelConfig | None = None) -> None:
    validate_base_lineage_integrity(source_dir)

    if cfg:
        validate_configs_match(source_dir, cfg)