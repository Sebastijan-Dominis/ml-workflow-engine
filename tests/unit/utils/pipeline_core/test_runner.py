"""Unit tests for pipeline core runner/step orchestration behavior."""

import pytest
from ml.utils.pipeline_core.runner import PipelineRunner
from ml.utils.pipeline_core.step import PipelineStep

pytestmark = pytest.mark.unit


class _AppendStep(PipelineStep[dict]):
    """Test step that appends events and increments `value`."""
    def __init__(self, name: str):
        """Store the step name used in emitted events."""
        self.name = name

    def before(self, ctx: dict) -> None:
        """Append a before-hook event to the context."""
        ctx["events"].append(f"before:{self.name}")

    def run(self, ctx: dict) -> dict:
        """Append a run event and increment the context value."""
        ctx["events"].append(f"run:{self.name}")
        ctx["value"] += 1
        return ctx

    def after(self, ctx: dict) -> None:
        """Append an after-hook event to the context."""
        ctx["events"].append(f"after:{self.name}")


class _ReplaceContextStep(PipelineStep[dict]):
    """Test step that replaces the input context object."""
    name = "replace"

    def run(self, ctx: dict) -> dict:
        """Return a new context with updated events and value."""
        # Return a new context object to validate that runner forwards it.
        return {"events": [*ctx["events"], "run:replace"], "value": ctx["value"] + 10}


class _NoHookStep(PipelineStep[dict]):
    """Test step that implements only `run`."""
    name = "nohook"

    def run(self, ctx: dict) -> dict:
        """Append a run event without custom hooks."""
        ctx["events"].append("run:nohook")
        return ctx


class _FailingStep(PipelineStep[dict]):
    """Test step that raises during execution."""
    name = "failing"

    def run(self, ctx: dict) -> dict:
        """Append a run event and then raise `RuntimeError`."""
        ctx["events"].append("run:failing")
        raise RuntimeError("boom")


class _TailStep(PipelineStep[dict]):
    """Test step used to verify execution stops after failures."""
    name = "tail"

    def run(self, ctx: dict) -> dict:
        """Append a run event to the context."""
        ctx["events"].append("run:tail")
        return ctx


def test_pipeline_runner_executes_before_run_after_in_order() -> None:
    """Verify hook execution order and in-place context updates."""
    runner: PipelineRunner = PipelineRunner([_AppendStep("first"), _AppendStep("second")])
    ctx = {"events": [], "value": 0}

    result = runner.run(ctx)

    assert result is ctx
    assert result["value"] == 2
    assert result["events"] == [
        "before:first",
        "run:first",
        "after:first",
        "before:second",
        "run:second",
        "after:second",
    ]


def test_pipeline_runner_propagates_replaced_context_between_steps() -> None:
    """Verify that replaced context objects are propagated to later steps."""
    runner: PipelineRunner = PipelineRunner([_ReplaceContextStep(), _AppendStep("next")])
    initial = {"events": [], "value": 1}

    result = runner.run(initial)

    assert result is not initial
    assert result["value"] == 12
    assert result["events"] == [
        "run:replace",
        "before:next",
        "run:next",
        "after:next",
    ]


def test_pipeline_runner_returns_input_context_when_no_steps() -> None:
    """Verify that an empty runner returns the input context unchanged."""
    runner: PipelineRunner = PipelineRunner([])
    ctx = {"events": [], "value": 5}

    result = runner.run(ctx)

    assert result is ctx
    assert result == {"events": [], "value": 5}


def test_pipeline_runner_uses_default_noop_hooks_from_base_step() -> None:
    """Verify that default no-op hooks do not interfere with `run` execution."""
    runner: PipelineRunner = PipelineRunner([_NoHookStep()])
    ctx = {"events": [], "value": 0}

    result = runner.run(ctx)

    assert result["events"] == ["run:nohook"]


def test_pipeline_runner_stops_execution_after_failing_step() -> None:
    """Verify that runner execution stops and propagates on step failure."""
    runner: PipelineRunner = PipelineRunner([_FailingStep(), _TailStep()])
    ctx = {"events": [], "value": 0}

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(ctx)

    assert ctx["events"] == ["run:failing"]
