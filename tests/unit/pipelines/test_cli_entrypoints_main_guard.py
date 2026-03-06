"""Unit tests for CLI entrypoint `__main__` guards in pipeline modules."""

from __future__ import annotations

import runpy
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("module_name", "argv"),
    [
        ("pipelines.data.build_interim_dataset", ["build_interim_dataset", "--help"]),
        ("pipelines.data.build_processed_dataset", ["build_processed_dataset", "--help"]),
        ("pipelines.data.register_raw_snapshot", ["register_raw_snapshot", "--help"]),
        (
            "pipelines.orchestration.data.execute_all_data_preprocessing",
            ["execute_all_data_preprocessing", "--help"],
        ),
        (
            "pipelines.orchestration.experiments.execute_all_experiments_with_latest",
            ["execute_all_experiments_with_latest", "--help"],
        ),
        (
            "pipelines.orchestration.experiments.execute_experiment_with_latest",
            ["execute_experiment_with_latest", "--help"],
        ),
        (
            "pipelines.orchestration.features.freeze_all_feature_sets",
            ["freeze_all_feature_sets", "--help"],
        ),
        ("pipelines.orchestration.master.run_all_workflows", ["run_all_workflows", "--help"]),
        ("pipelines.promotion.promote", ["promote", "--help"]),
        ("pipelines.runners.evaluate", ["evaluate", "--help"]),
        ("pipelines.runners.explain", ["explain", "--help"]),
        ("pipelines.runners.train", ["train", "--help"]),
        ("pipelines.search.search", ["search", "--help"]),
    ],
)
def test_cli_module_main_guard_executes_and_exits_on_help(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    argv: list[str],
) -> None:
    """Execute module as `__main__` and assert help-path exits cleanly with code zero."""
    monkeypatch.setattr(sys, "argv", argv)
    sys.modules.pop(module_name, None)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(module_name, run_name="__main__")

    assert exc_info.value.code == 0
