"""Application service orchestrating end-to-end promotion workflow."""

import logging
from contextlib import contextmanager

from filelock import FileLock

from ml.exceptions import UserError
from ml.promotion.context import PromotionContext
from ml.promotion.getters.get import get_runners_metadata
from ml.promotion.persister import PromotionPersister
from ml.promotion.state_loader import PromotionStateLoader
from ml.promotion.strategies.production import ProductionPromotionStrategy
from ml.promotion.strategies.staging import StagingPromotionStrategy
from ml.promotion.validation.validate import (
    validate_explainability_artifacts, validate_run_dirs, validate_run_ids)

logger = logging.getLogger(__name__)

class PromotionService:
    """Coordinates validation, state loading, strategy execution, and persistence."""

    def __init__(self):
        """Initialize promotion service dependencies."""
        self._state_loader = PromotionStateLoader()
        self._persister = PromotionPersister()

    def run(self, context: PromotionContext):
        """Execute complete promotion workflow for the given context.

        Args:
            context: Promotion execution context.

        Returns:
            Any: Strategy result payload persisted by the promotion service.
        """

        context = self._validate(context)

        with self._registry_lock(context):
            state = self._state_loader.load(context)
            strategy = self._get_strategy(context.args.stage)
            result = strategy.execute(context, state)
            self._persister.persist(context, state, result)

        return result

    def _validate(self, context: PromotionContext) -> PromotionContext:
        """Validate input run artifacts and enrich context with runner metadata.

        Args:
            context: Promotion execution context.

        Returns:
            PromotionContext: Enriched and validated promotion context.
        """

        validate_run_dirs(context.paths.train_run_dir, context.paths.eval_run_dir, context.paths.explain_run_dir)
        runners_metadata = get_runners_metadata(context.paths.train_run_dir, context.paths.eval_run_dir, context.paths.explain_run_dir)
        validate_run_ids(args=context.args, runners_metadata=runners_metadata)
        validate_explainability_artifacts(runners_metadata=runners_metadata, args=context.args)
        context.runners_metadata = runners_metadata
        return context
    
    @contextmanager
    def _registry_lock(self, context: PromotionContext):
        """Acquire file lock around registry mutations to avoid concurrent writes.

        Args:
            context: Promotion execution context containing registry paths.

        Yields:
            None: Context manager yielding lock scope.
        """

        lock_path = str(context.paths.registry_path) + ".lock"
        lock = FileLock(lock_path, timeout=300)
        with lock:
            yield
    
    def _get_strategy(self, stage: str):
        """Resolve stage-specific promotion strategy implementation.

        Args:
            stage: Target promotion stage.

        Returns:
            Any: Instantiated promotion strategy implementation.
        """

        if stage == "production":
            return ProductionPromotionStrategy()
        elif stage == "staging":
            return StagingPromotionStrategy()
        else:
            msg = f"Unknown stage specified: {stage}. Supported stages are 'production' and 'staging'."
            logger.error(msg)
            raise UserError(msg)
    