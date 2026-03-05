"""Unit tests for pipeline core runner/step orchestration behavior."""

import pytest
from ml.utils.pipeline_core.runner import PipelineRunner
from ml.utils.pipeline_core.step import PipelineStep

pytestmark = pytest.mark.unit


class _AppendStep(PipelineStep[dict]):
    """Test step that appends its name to an 'events' list in the context and increments a 'value' field."""
    def __init__(self, name: str):
        """Initialize step with a name to append to events.

        Args:
            name: Name of the step to append to events.
        """
        self.name = name

    def before(self, ctx: dict) -> None:
        """Append a 'before' event to the context's events list.

        Args:
            ctx: Pipeline context dictionary with an 'events' list.

        Returns:
            None: Modifies context in place.
        """
        ctx["events"].append(f"before:{self.name}")

    def run(self, ctx: dict) -> dict:
        """Append a 'run' event to the context's events list and increment the 'value' field.

        Args:
            ctx: Pipeline context dictionary with an 'events' list and a 'value' field.

        Returns:
            dict: Updated context with new events and incremented value.
        """
        ctx["events"].append(f"run:{self.name}")
        ctx["value"] += 1
        return ctx

    def after(self, ctx: dict) -> None:
        """Append an 'after' event to the context's events list.

        Args:
            ctx: Pipeline context dictionary with an 'events' list.
        Returns:
            None: Modifies context in place.
        """
        ctx["events"].append(f"after:{self.name}")


class _ReplaceContextStep(PipelineStep[dict]):
    """Test step that replaces the entire context with a new dictionary in its run method."""
    name = "replace"

    def run(self, ctx: dict) -> dict:
        """Replace the input context with a new dictionary containing updated events and value.

        Args:
            ctx: Pipeline context dictionary with an 'events' list and a 'value' field.

        Returns:
            dict: A new context dictionary with updated events and value.
        """
        # Return a new context object to validate that runner forwards it.
        return {"events": [*ctx["events"], "run:replace"], "value": ctx["value"] + 10}


class _NoHookStep(PipelineStep[dict]):
    """Test step that does not implement before or after hooks, only a run method."""
    name = "nohook"

    def run(self, ctx: dict) -> dict:
        """Append a 'run' event to the context's events list without implementing before or after hooks.

        Args:
            ctx: Pipeline context dictionary with an 'events' list.

        Returns:
            dict: Updated context with a new run event.
        """
        ctx["events"].append("run:nohook")
        return ctx


class _FailingStep(PipelineStep[dict]):
    """Test step that raises an exception in its run method to validate that the runner stops execution and propagates the error."""
    name = "failing"

    def run(self, ctx: dict) -> dict:
        """Append a 'run' event to the context's events list and then raise a RuntimeError.

        Args:
            ctx: Pipeline context dictionary with an 'events' list.

        Returns:
            dict: This method does not return a context as it raises an exception.
        """
        ctx["events"].append("run:failing")
        raise RuntimeError("boom")


class _TailStep(PipelineStep[dict]):
    """Test step that appends a 'run' event to the context's events list, used to validate that execution does not reach this step if a previous step fails."""
    name = "tail"

    def run(self, ctx: dict) -> dict:
        """Append a 'run' event to the context's events list.

        Args:
            ctx: Pipeline context dictionary with an 'events' list.

        Returns:
            dict: Updated context with a new run event.
        """
        ctx["events"].append("run:tail")
        return ctx


def test_pipeline_runner_executes_before_run_after_in_order() -> None:
    """Test that the PipelineRunner executes before, run, and after hooks of each step in the correct order, and that the context is correctly passed and modified through the steps."""
    runner = PipelineRunner([_AppendStep("first"), _AppendStep("second")])
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
    """Test that if a step returns a completely new context object, the PipelineRunner correctly forwards this new context to subsequent steps in the pipeline, and that the before and after hooks of subsequent steps operate on the updated context."""
    runner = PipelineRunner([_ReplaceContextStep(), _AppendStep("next")])
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
    """Test that if the PipelineRunner is initialized with an empty list of steps, it returns the input context unchanged when run."""
    runner = PipelineRunner([])
    ctx = {"events": [], "value": 5}

    result = runner.run(ctx)

    assert result is ctx
    assert result == {"events": [], "value": 5}


def test_pipeline_runner_uses_default_noop_hooks_from_base_step() -> None:
    """Test that if a step does not implement before or after hooks, the PipelineRunner still executes the run method of the step and that the default no-op implementations of before and after from the base PipelineStep class do not interfere with execution. This validates that steps can choose to only implement the run method if they do not need before/after behavior.
    """
    runner = PipelineRunner([_NoHookStep()])
    ctx = {"events": [], "value": 0}

    result = runner.run(ctx)

    assert result["events"] == ["run:nohook"]


def test_pipeline_runner_stops_execution_after_failing_step() -> None:
    """Test that if a step raises an exception during its run method, the PipelineRunner stops execution of subsequent steps and propagates the error, and that any steps after the failing step do not have their hooks executed."""
    runner = PipelineRunner([_FailingStep(), _TailStep()])
    ctx = {"events": [], "value": 0}

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(ctx)

    assert ctx["events"] == ["run:failing"]
