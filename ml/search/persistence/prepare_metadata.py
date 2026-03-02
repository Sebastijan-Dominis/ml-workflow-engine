"""Metadata payload construction for search experiment persistence."""

import logging
from dataclasses import asdict
from pathlib import Path

from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.registry.tabular_splits import AllSplitsInfo
from ml.utils.git import get_git_commit
from ml.utils.formatting.iso_no_col import iso_no_colon

logger = logging.getLogger(__name__)

def prepare_metadata(
    model_cfg: SearchModelConfig, 
    *, 
    search_results: dict, 
    owner: str, 
    experiment_id: str, 
    timestamp: str, 
    feature_lineage: list[dict], 
    pipeline_hash: str,
    scoring_method: str,
    splits_info: AllSplitsInfo
) -> dict:
    """Build metadata record for a completed search experiment.

    Args:
        model_cfg: Validated search model configuration.
        search_results: Search results payload to persist.
        owner: Owner identifier for the search run.
        experiment_id: Experiment identifier.
        timestamp: Run creation timestamp.
        feature_lineage: Feature lineage records used by the run.
        pipeline_hash: Pipeline configuration hash.
        scoring_method: Metric/scoring method used during search.
        splits_info: Dataset split information for reproducibility metadata.

    Returns:
        Search metadata record ready for persistence.

    Notes:
        Datetime lineage fields are normalized to ISO strings before payload
        assembly to keep metadata JSON-serializable.

    Side Effects:
        Reads git commit metadata from the active repository context.
    """

    # Convert search_lineage.created_at and model_specs_lineage.created_at from datetime to ISO format string to avoid errors during JSON serialization in save_metadata
    search_created_at_str = iso_no_colon(model_cfg.search_lineage.created_at)
    model_specs_created_at_str = iso_no_colon(model_cfg.model_specs_lineage.created_at)
    model_cfg_dict = model_cfg.model_dump(by_alias=True)
    model_cfg_dict["search_lineage"]["created_at"] = search_created_at_str
    model_cfg_dict["model_specs_lineage"]["created_at"] = model_specs_created_at_str

    problem = model_cfg.problem
    segment = model_cfg.segment.name
    version = model_cfg.version

    algorithm = model_cfg.algorithm.value if getattr(model_cfg, "algorithm", None) else None
    seed = model_cfg.seed or "none"
    hardware = (
        model_cfg.search.hardware.model_dump()
        if getattr(model_cfg.search, "hardware", None)
        else {}
    )

    meta = model_cfg.meta
    sources = meta.sources or {}
    env = meta.env or "default"
    best_params_path = meta.best_params_path or "none"

    pipeline_version = getattr(model_cfg.pipeline, "version", "none")

    git_commit = get_git_commit(Path("."))
    config_hash = meta.config_hash or "none"
    validation_status = meta.validation_status or "unknown"
    splits_info_dict = asdict(splits_info)

    record = {
        "metadata": {
            "problem": problem,
            "segment": segment,
            "version": version,
            "experiment_id": experiment_id,
            "sources": sources,
            "env": env,
            "best_params_path": best_params_path,
            "algorithm": algorithm,
            "pipeline_version": pipeline_version,
            "created_by": "search.py",
            "created_at": timestamp,
            "owner": owner,
            "feature_lineage": feature_lineage,
            "seed": seed,
            "hardware": hardware,
            "git_commit": git_commit,
            "config_hash": config_hash,
            "validation_status": validation_status,
            "pipeline_hash": pipeline_hash,
            "scoring_method": scoring_method,
            "splits_info": splits_info_dict
        },
        "config": model_cfg_dict,
        "search_results": search_results,
    }

    task_type = model_cfg.task.type

    if task_type == "regression":
        transform = model_cfg.target.transform
        record["metadata"]["target_transform"] = {
            "enabled": transform.enabled,
            **({"type": transform.type} if transform.type is not None else {}),
            **({"lambda": transform.lambda_value} if transform.lambda_value is not None else {}),
        }

    elif task_type == "classification":
        record["metadata"]["class_weighting"] = model_cfg.class_weighting.model_dump()

    return record
