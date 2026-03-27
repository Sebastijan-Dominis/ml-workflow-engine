"""A module for loading model registry information in post-promotion pipelines."""
import argparse
import logging
from pathlib import Path

from ml.exceptions import PipelineContractError
from ml.post_promotion.shared.classes.function_returns import ModelRegistryInfo
from ml.promotion.validation.registry_entry import validate_registry_entry
from ml.utils.loaders import load_yaml

logger = logging.getLogger(__name__)

def get_model_registry_info(args: argparse.Namespace) -> ModelRegistryInfo:
    """Fetch model registry information for the latest snapshot.

    Returns:
        ModelRegistryInfo: An instance of ModelRegistryInfo containing model registry information.
    """
    model_registry = Path("model_registry/models.yaml")
    registry = load_yaml(model_registry)

    entry = registry.get(args.problem, {}).get(args.segment, {})

    prod_meta_raw = entry.get("production")
    stage_meta_raw = entry.get("staging")

    if not prod_meta_raw and not stage_meta_raw:
        msg = f"No production or staging model found in registry for problem '{args.problem}' and segment '{args.segment}'."
        logger.error(msg)
        raise PipelineContractError(msg)

    prod_meta, stage_meta = None, None
    if prod_meta_raw:
        prod_meta = validate_registry_entry(prod_meta_raw)
    if stage_meta_raw:
        stage_meta = validate_registry_entry(stage_meta_raw)

    return ModelRegistryInfo(prod_meta=prod_meta, stage_meta=stage_meta)
