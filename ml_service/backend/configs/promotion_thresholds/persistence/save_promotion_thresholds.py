"""A module for saving promotion thresholds configuration."""
import copy
from pathlib import Path

from ml.promotion.config.promotion_thresholds import PromotionThresholds

from ml_service.backend.configs.persistence.save_config import save_config


def save_promotion_thresholds(
    *,
    thresholds: dict,
    validated: PromotionThresholds,
    config_path: Path,
    problem_type: str,
    segment: str
) -> None:
    """Save the promotion thresholds to the specified path, ensuring atomic write and handling errors.

    Args:
        thresholds (dict): The existing thresholds dictionary loaded from the config file.
        validated (PromotionThresholds): The validated promotion thresholds data to be saved.
        config_path (Path): The file path where the thresholds should be saved.
        problem_type (str): The problem type for which the thresholds are being saved.
        segment (str): The segment for which the thresholds are being saved.
    """
    thresholds_new = copy.deepcopy(thresholds) if thresholds else {}

    thresholds_new.setdefault(problem_type, {})[segment] = validated.model_dump(mode="json")

    save_config(thresholds_new, config_path)
