from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from ml.utils.pipeline_core.runner import PipelineRunner
from ml.utils.pipeline_core.step import PipelineStep

pytestmark = pytest.mark.integration


@dataclass
class DummyStep(PipelineStep[dict[str, Any]]):
    name: str

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = dict(ctx)
        ctx[self.name] = ctx.get(self.name, 0) + 1
        return ctx


def test_pipeline_runner_executes_steps_in_order() -> None:
    steps: list[PipelineStep[dict[str, Any]]] = [
        DummyStep(name="a"),
        DummyStep(name="b"),
        DummyStep(name="a"),
    ]
    runner = PipelineRunner(steps=steps)
    ctx = {"a": 0}
    out = runner.run(ctx)

    assert out["a"] == 2
    assert out["b"] == 1
