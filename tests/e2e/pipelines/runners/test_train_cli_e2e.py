"""E2E-style smoke tests for the training CLI boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pipelines.runners import train as train_module

pytestmark = pytest.mark.e2e


def test_train_main_executes_end_to_end_control_flow_with_cli_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise parse_args + main with realistic argv and fully successful execution."""
    experiment_dir = tmp_path / "experiments" / "no_show" / "global" / "v1" / "exp_100"
    (experiment_dir / "search").mkdir(parents=True)

    monkeypatch.setattr(
        train_module.sys,
        "argv",
        [
            "train",
            "--problem",
            "no_show",
            "--segment",
            "global",
            "--version",
            "v1",
            "--env",
            "test",
            "--strict",
            "true",
            "--experiment-id",
            "exp_100",
            "--snapshot-binding-key",
            "snapshot_100",
            "--logging-level",
            "INFO",
            "--clean-up-failure-management",
            "false",
            "--overwrite-existing",
            "false",
        ],
    )

    monkeypatch.setattr(train_module, "iso_no_colon", lambda _dt: "20260306T200000")
    monkeypatch.setattr(train_module, "uuid4", lambda: SimpleNamespace(hex="1234567890abcdef"))
    monkeypatch.setattr(train_module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(train_module, "add_file_handler", lambda path, level: None)
    monkeypatch.setattr(train_module, "get_snapshot_path", lambda snapshot_id, parent_dir: experiment_dir)

    model_cfg = SimpleNamespace(algorithm=SimpleNamespace(value="CatBoost"))
    monkeypatch.setattr(train_module, "load_and_validate_config", lambda *args, **kwargs: model_cfg)
    monkeypatch.setattr(train_module, "add_config_hash", lambda cfg: cfg)
    monkeypatch.setattr(train_module, "validate_lineage_integrity", lambda search_dir: None)
    monkeypatch.setattr(train_module, "validate_reproducibility", lambda runtime_path: None)
    monkeypatch.setattr(train_module, "validate_logical_config", lambda model_cfg, search_dir: None)
    monkeypatch.setattr(train_module, "validate_pipeline_cfg", lambda metadata_path, model_cfg: None)

    training_output = SimpleNamespace(
        model=object(),
        pipeline=None,
        pipeline_cfg_hash=None,
        lineage=[],
        metrics={"auc": 0.81},
    )
    trainer = SimpleNamespace(train=lambda *args, **kwargs: training_output)
    monkeypatch.setattr(train_module, "get_trainer", lambda algorithm: trainer)

    monkeypatch.setattr(
        train_module,
        "save_model",
        lambda model, train_run_dir: train_run_dir / "model.cbm",
    )
    monkeypatch.setattr(train_module, "hash_artifact", lambda artifact_path: "artifact_hash")

    persisted: dict[str, object] = {}
    monkeypatch.setattr(
        train_module,
        "persist_training_run",
        lambda model_cfg, **kwargs: persisted.update(kwargs),
    )

    cleanup_calls: list[tuple[Path, bool, str]] = []
    monkeypatch.setattr(
        train_module,
        "delete_failure_management_folder",
        lambda *, folder_path, cleanup, stage: cleanup_calls.append((folder_path, cleanup, stage)),
    )

    code = train_module.main()

    assert code == 0
    assert persisted["metrics"] == {"auc": 0.81}
    assert persisted["train_run_id"] == "20260306T200000_12345678"
    assert cleanup_calls[0][1:] == (False, "train")

    # The following lines of code can be dangerous if the function implementation changes unexpectedly, since they call the actual function. I included it here, since the test leaves empty folders within the actual failure management directory. I use the snippet below to get rid of this harmless, but annoying side-effect of this test. You can manually delete those folders if you want to as well, and the code guarantees that no experiments or training runs will ever have those same IDs, so there's no side effects apart from it being annoying.
    # There are three other tests like this around the repo. Alltogether, they leave folders called "exp_2", "exp_3", "exp_5" and "exp_100" within `failure_management`. This comment should help you understand what is happening and deal with it.
    # I decided to comment the code below out for safety reasons, but leave it in place in case you want to re-enable it for cleanup purposes (I use it locally). Just be mindful of the implications.
    # As of now, I have not yet been able to find time to deal with this issue directly - it's just a nuisance anyway. Of course, you are free to fix it if you find a solution before I do!
    # - Sebastijan

    # leftover_dir = Path("failure_management") / "exp_100" / "training" / "20260306T200000_12345678"
    # delete_failure_management_folder(
    #     folder_path=leftover_dir,
    #     cleanup=True,
    #     stage="train"
    # )
