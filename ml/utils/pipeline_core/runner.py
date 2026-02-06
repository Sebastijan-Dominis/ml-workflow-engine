import logging
logger = logging.getLogger(__name__)
from ml.utils.pipeline_core.step import PipelineStep
from typing import TypeVar, Generic


C = TypeVar("C")

class PipelineRunner(Generic[C]):
    def __init__(self, steps: list[PipelineStep[C]]):
        self.steps = steps

    def run(self, ctx: C) -> C:
        for step in self.steps:
            if hasattr(step, "before"):
                step.before(ctx)
            ctx = step.run(ctx)
            if hasattr(step, "after"):
                step.after(ctx)
        return ctx
