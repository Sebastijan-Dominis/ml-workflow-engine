"""Registry update and diff-persistence helpers for promotion workflow."""

import copy
import logging
import os
from pathlib import Path

import yaml
from ml.exceptions import PersistenceError
from ml.promotion.constants.constants import Stage

logger = logging.getLogger(__name__)

def update_registry_and_archive(
    *,
    model_registry: dict,
    archive_registry: dict,
    stage: Stage,
    run_info: dict,
    problem: str,
    segment: str,
    registry_path: Path = Path("model_registry") / "models.yaml",
    archive_path: Path = Path("model_registry") / "archive.yaml"
) -> dict:
    """Update active registry and archive previous production entry when needed.

    Args:
        model_registry: Current active model registry.
        archive_registry: Current archive registry.
        stage: Promotion stage.
        run_info: Registry run-info payload.
        problem: Problem key.
        segment: Segment key.
        registry_path: Active registry file path.
        archive_path: Archive registry file path.

    Returns:
        dict: Updated active registry dictionary.
    """

    new_registry = copy.deepcopy(model_registry, {})

    new_registry.setdefault(problem, {})
    new_registry[problem].setdefault(segment, {})
    new_archive = None

    if stage == "production":
        current_prod_model_info = new_registry[problem][segment].get("production")
        if current_prod_model_info:
            promotion_id = current_prod_model_info.get("promotion_id")
            if not promotion_id:
                msg = f"Current production model for problem '{problem}' and segment '{segment}' is missing promotion_id. Cannot archive without promotion_id. Current production model info: {current_prod_model_info}"
                logger.error(msg)
                raise PersistenceError(msg)
            new_archive = copy.deepcopy(archive_registry, {})
            new_archive.setdefault(problem, {}).setdefault(segment, {})
            new_archive[problem][segment][promotion_id] = current_prod_model_info

        new_registry[problem][segment]["production"] = run_info
    else:
        # Only one staging model is allowed per problem/segment, so we can directly set it without archiving
        new_registry[problem][segment]["staging"] = run_info

    try:
        if new_archive is not None:
            temp_archive_path = archive_path.with_suffix(".tmp")
            with open(temp_archive_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(new_archive, f, sort_keys=False)
            os.replace(temp_archive_path, archive_path)

        temp_registry_path = registry_path.with_suffix(".tmp")
        with open(temp_registry_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(new_registry, f, sort_keys=False)
        os.replace(temp_registry_path, registry_path)

        return new_registry
    except Exception as e:
        msg = f"Failed to update model registry and archive. Run info: {run_info}"
        logger.exception(msg)
        raise PersistenceError(msg) from e

def persist_registry_diff(
    *,
    previous_registry: dict,
    updated_registry: dict,
    run_dir: Path
) -> None:
    """Persist before/after registry snapshot for auditability.

    Args:
        previous_registry: Registry state before update.
        updated_registry: Registry state after update.
        run_dir: Promotion run directory.

    Returns:
        None: Writes registry-diff artifact to disk.
    """

    diff_path = run_dir / "registry_diff.yaml"
    diff = {
        "previous": previous_registry,
        "updated": updated_registry
    }
    try:
        with open(diff_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(diff, f, sort_keys=False)
    except Exception as e:
        msg = f"Failed to persist registry diff to {diff_path}"
        logger.exception(msg)
        raise PersistenceError(msg) from e
