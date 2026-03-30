"""Integration tests for the pipeline runner utilities.

These tests ensure `PipelineRunner` executes step `before`, `run` and `after`
hooks in order and returns the final context object.
"""

from __future__ import annotations

from ml.utils.pipeline_core.runner import PipelineRunner
from ml.utils.pipeline_core.step import PipelineStep


class IncStep(PipelineStep[dict[str, int]]):
    """A tiny step that increments an integer counter in the context."""

    def __init__(self, key: str, amount: int) -> None:
        self.key = key
        self.amount = amount

    def before(self, ctx: dict[str, int]) -> None:
        ctx.setdefault(self.key, 0)

    def run(self, ctx: dict[str, int]) -> dict[str, int]:
        ctx[self.key] += self.amount
        return ctx

    def after(self, ctx: dict[str, int]) -> None:
        ctx[f"after_{self.key}"] = 1


def test_pipeline_runner_executes_hooks_and_steps() -> None:
    """PipelineRunner runs steps in order and invokes hooks appropriately."""

    steps: list[PipelineStep[dict[str, int]]] = [IncStep("a", 1), IncStep("a", 2), IncStep("b", 3)]
    runner: PipelineRunner[dict[str, int]] = PipelineRunner(steps)
    ctx: dict[str, int] = {}
    res = runner.run(ctx)

    assert res["a"] == 3
    assert res["b"] == 3
    assert res["after_a"] == 1
    assert res["after_b"] == 1
