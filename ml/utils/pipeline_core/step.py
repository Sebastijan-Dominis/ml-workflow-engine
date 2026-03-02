"""Abstract step contract for context-driven pipeline execution."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ContextT = TypeVar("ContextT")

class PipelineStep(ABC, Generic[ContextT]):
    """Base class for pipeline steps operating on a shared context object."""

    name: str = "unnamed"

    def before(self, ctx: ContextT) -> None:
        """Optional hook executed before `run`; default implementation is no-op.

        Args:
            ctx: Pipeline context object.

        Returns:
            None: No-op by default.
        """

        ...

    def after(self, ctx: ContextT) -> None:
        """Optional hook executed after `run`; default implementation is no-op.

        Args:
            ctx: Pipeline context object.

        Returns:
            None: No-op by default.
        """

        ...

    @abstractmethod
    def run(self, ctx: ContextT) -> ContextT:
        """Execute step logic and return the updated context.

        Args:
            ctx: Pipeline context object.

        Returns:
            ContextT: Updated context object.
        """

        ...
