from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ContextT = TypeVar("ContextT")

class PipelineStep(ABC, Generic[ContextT]):
    name: str = "unnamed"

    def before(self, ctx: ContextT) -> None:
        ...

    def after(self, ctx: ContextT) -> None:
        ...

    @abstractmethod
    def run(self, ctx: ContextT) -> ContextT:
        ...
