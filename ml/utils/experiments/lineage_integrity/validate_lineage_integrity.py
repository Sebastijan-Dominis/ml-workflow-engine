from pathlib import Path

from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.utils.experiments.lineage_integrity.validations.base import validate_base_lineage_integrity
from ml.utils.experiments.lineage_integrity.validations.configs_match import validate_configs_match


def validate_lineage_integrity(experiment_dir: Path, train_dir: Path | None = None, cfg: TrainModelConfig | SearchModelConfig | None = None) -> None:
    validate_base_lineage_integrity(experiment_dir)

    if train_dir and cfg:
        validate_configs_match(train_dir, cfg)