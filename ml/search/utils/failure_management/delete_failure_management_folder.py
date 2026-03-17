"""Cleanup helpers for search/training failure-management directories."""

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

DIRS_OK_TO_DELETE = ["learn", "test", "tmp"]

def delete_failure_management_folder(
    *,
    folder_path: Path,
    cleanup: bool,
    stage: Literal["search", "train"]
) -> None:
    """Safely remove failure-management directories when cleanup is enabled."""

    logger.debug(f"Running delete_failure_management_folder with folder_path={folder_path} and cleanup={cleanup}")
    if not cleanup:
        logger.info(f"Skipping cleanup of failure management folder for experiment {folder_path.name}.")
        return

    if folder_path.exists() and folder_path.is_dir():
        entries = list(folder_path.iterdir())

        subdirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        if subdirs:
            unexpected_subdirs = [d.name for d in subdirs if d.name not in DIRS_OK_TO_DELETE]
            if unexpected_subdirs:
                logger.warning(
                    f"Failure management folder {folder_path} contains unexpected subdirectories "
                    f"({unexpected_subdirs}). Skipping deletion for safety."
                )
                return
            logger.warning(
                f"Failure management folder {folder_path} contains subdirectories marked ok to delete "
                f"({[d.name for d in subdirs]}). These will be deleted along with their contents."
            )

        for file in files:
            file.unlink()

        for subdir in subdirs:
            for item in subdir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    logger.warning(
                        f"Found unexpected nested directory {item} inside {subdir}. Terminating the deletion process for safety."
                    )
                    return
            subdir.rmdir()

        folder_path.rmdir()

        run_name = "experiment" if stage == "search" else "training"
        logger.info(f"Successfully deleted failure management folder for {run_name} {folder_path.name} at {folder_path}.")
    if stage == "search":
        if stage == "search":
            parent = folder_path.parent
            if parent.exists() and parent.is_dir() and not any(parent.iterdir()):
                try:
                    parent.rmdir()
                    logger.info(
                        "Deleted the main failure management directory, as it is now empty: %s.",
                        parent.name,
                    )
                except OSError:
                    logger.debug(
                        "Skipping deletion of %s (possibly a Docker mount or busy directory).",
                        parent,
                    )
    elif stage == "train":

        training_failure_management_dir = folder_path.parent
        if training_failure_management_dir.exists() and training_failure_management_dir.is_dir() and not any(training_failure_management_dir.iterdir()):
            try:
                training_failure_management_dir.rmdir()
                logger.info(f"Deleted the training failure management directory, as it is now empty: {training_failure_management_dir.name}.")
            except OSError:
                logger.debug(f"Skipping deletion of {training_failure_management_dir} (possibly a Docker mount or busy directory).")

        experiment_failure_management_dir = training_failure_management_dir.parent
        if experiment_failure_management_dir.exists() and experiment_failure_management_dir.is_dir() and not any(experiment_failure_management_dir.iterdir()):
            try:
                experiment_failure_management_dir.rmdir()
                logger.info(f"Deleted the experiment failure management directory, as it is now empty: {experiment_failure_management_dir.name}.")
            except OSError:
                logger.debug(f"Skipping deletion of {experiment_failure_management_dir} (possibly a Docker mount or busy directory).")

        main_failure_management_dir = experiment_failure_management_dir.parent
        if main_failure_management_dir.exists() and main_failure_management_dir.is_dir() and not any(main_failure_management_dir.iterdir()):
            try:
                main_failure_management_dir.rmdir()
                logger.info(f"Deleted the main failure management directory, as it is now empty: {main_failure_management_dir.name}.")
            except OSError:
                logger.debug(f"Skipping deletion of {main_failure_management_dir} (possibly a Docker mount or busy directory).")
