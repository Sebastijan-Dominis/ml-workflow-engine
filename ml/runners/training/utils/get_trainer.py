import logging

from ml.exceptions import PipelineContractError
from ml.registry.train_registry import TRAINERS
from ml.runners.training.trainers.base import Trainer

logger = logging.getLogger(__name__)

def get_trainer(key: str) -> Trainer:
    trainer_cls = TRAINERS.get(key)

    if not trainer_cls:
        msg = f"No trainer found for algorithm '{key}'."
        logger.error(msg)
        raise PipelineContractError(msg)
    
    trainer = trainer_cls()

    logger.debug(
        "Using trainer %s for algorithm=%s",
        trainer.__class__.__name__,
        key,
    )

    return trainer