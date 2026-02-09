import logging
from pathlib import Path
from typing import overload

from ml.config.hashing import compute_config_hash
from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import PipelineContractError
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

@overload
def validate_configs_match(train_dir: Path, cfg: TrainModelConfig) -> None: ...

@overload
def validate_configs_match(train_dir: Path, cfg: SearchModelConfig) -> None: ...

def validate_configs_match(train_dir: Path, cfg: TrainModelConfig | SearchModelConfig) -> None:
    train_metadata_path = train_dir / "metadata.json"
    if not train_metadata_path.exists():
        msg = f"Lineage integrity validation failed: {train_metadata_path} does not exist."
        logger.error(msg)
        raise PipelineContractError(msg)
    train_metadata = load_json(train_metadata_path)

    runtime_dict = cfg.model_dump(exclude={"_meta"}, by_alias=True)
    config_hash = compute_config_hash(runtime_dict)

    if train_metadata.get("config_hash") != config_hash:
        msg = (
            f"Lineage integrity validation failed: config hash mismatch.\n"
            f"Expected hash: {train_metadata.get('config_hash')}\n"
            f"Actual hash: {config_hash}\n"
            f"Train metadata path: {train_metadata_path}"
        )
        logger.error(msg)
        raise PipelineContractError(msg)
    
    logger.debug("Lineage integrity validation passed: config hash matches.")