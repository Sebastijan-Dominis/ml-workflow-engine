"""Execution utilities for running ordered pipeline step sequences."""

import logging
from typing import Generic, TypeVar

from ml.utils.pipeline_core.step import PipelineStep

logger = logging.getLogger(__name__)

C = TypeVar("C")

class PipelineRunner(Generic[C]):
    """Generic runner that executes pipeline steps against a mutable context."""

    def __init__(self, steps: list[PipelineStep[C]]):
        """Initialize runner with an ordered collection of pipeline steps.

        Args:
            steps: Ordered pipeline steps.

        Returns:
            None: Initializes runner state.
        """

        self.steps = steps

    def run(self, ctx: C) -> C:
        """Execute all steps with optional `before`/`after` hooks and return context.

        Args:
            ctx: Mutable pipeline context.

        Returns:
            C: Final context after all steps run.
        """

        for step in self.steps:
            if hasattr(step, "before"):
                step.before(ctx)
            ctx = step.run(ctx)
            if hasattr(step, "after"):
                step.after(ctx)
        return ctx
