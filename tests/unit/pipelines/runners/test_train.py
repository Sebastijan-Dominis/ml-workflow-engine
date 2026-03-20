"""Unit tests for training runner CLI orchestration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ml.exceptions import PipelineContractError
from pipelines.runners import train as module

pytestmark = pytest.mark.unit


def test_parse_args_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse required args and preserve documented optional defaults."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["train", "--problem", "no_show", "--segment", "global", "--version", "v1"],
    )

    args = module.parse_args()

    assert args.problem == "no_show"
    assert args.segment == "global"
    assert args.version == "v1"
    assert args.train_run_id is None
    assert args.env == "default"
    assert args.strict is True
    assert args.experiment_id == "latest"
    assert args.logging_level == "INFO"
    assert args.clean_up_failure_management is True
    assert args.overwrite_existing is False


def test_main_returns_resolved_code_when_experiment_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map snapshot-resolution failures through the shared exit-code resolver."""
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            train_run_id=None,
            env="default",
            strict=True,
            experiment_id="latest",
            logging_level="INFO",
            clean_up_failure_management=True,
            overwrite_existing=False,
            snapshot_binding_key=None,
        ),
    )
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)

    err = RuntimeError("snapshot missing")
    monkeypatch.setattr(module, "get_snapshot_path", lambda snapshot_id, parent_dir: (_ for _ in ()).throw(err))
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 29 if e is err else 99)

    code = module.main()

    assert code == 29


def test_main_returns_one_when_existing_train_run_has_files_and_no_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject overwriting populated train run directory unless overwrite is enabled."""
    experiment_dir = tmp_path / "experiments" / "no_show" / "global" / "v1" / "exp_1"
    train_run_dir = experiment_dir / "training" / "run_1"
    train_run_dir.mkdir(parents=True)
    (train_run_dir / "model.pkl").write_text("existing", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            train_run_id="run_1",
            env="default",
            strict=True,
            experiment_id="exp_1",
            logging_level="INFO",
            clean_up_failure_management=True,
            overwrite_existing=False,
            snapshot_binding_key=None,
        ),
    )
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(module, "get_snapshot_path", lambda snapshot_id, parent_dir: experiment_dir)
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: None)

    code = module.main()

    assert code == 1


def test_main_runs_training_and_persists_outputs_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Execute full happy path and persist model, metrics, and metadata contracts."""
    experiment_dir = tmp_path / "experiments" / "no_show" / "global" / "v1" / "exp_2"
    (experiment_dir / "search").mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            train_run_id=None,
            env="test",
            strict=False,
            experiment_id="exp_2",
            logging_level="DEBUG",
            clean_up_failure_management=True,
            overwrite_existing=False,
            snapshot_binding_key=None,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T180000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: None)
    monkeypatch.setattr(module, "get_snapshot_path", lambda snapshot_id, parent_dir: experiment_dir)

    cfg = SimpleNamespace(algorithm=SimpleNamespace(value="CatBoost"))
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(module, "add_config_hash", lambda model_cfg: model_cfg)
    monkeypatch.setattr(module, "validate_lineage_integrity", lambda search_dir: None)
    monkeypatch.setattr(module, "validate_reproducibility", lambda runtime_path: None)
    monkeypatch.setattr(module, "validate_logical_config", lambda model_cfg, search_dir: None)
    monkeypatch.setattr(module, "validate_pipeline_cfg", lambda metadata_path, model_cfg: None)

    train_output = SimpleNamespace(
        model=object(),
        pipeline=None,
        pipeline_cfg_hash=None,
        lineage=[SimpleNamespace(feature="x")],
        metrics={"auc": 0.8},
    )
    trainer = SimpleNamespace(train=lambda *args, **kwargs: train_output)
    monkeypatch.setattr(module, "get_trainer", lambda algorithm: trainer)

    model_path = experiment_dir / "training" / "20260306T180000_abcdef01" / "model.pkl"
    monkeypatch.setattr(module, "save_model", lambda model, train_run_dir: model_path)
    monkeypatch.setattr(module, "hash_artifact", lambda artifact_path: "hash123")

    persisted: dict[str, Any] = {}
    monkeypatch.setattr(
        module,
        "persist_training_run",
        lambda model_cfg, **kwargs: persisted.update(kwargs),
    )

    cleanup_calls: list[tuple[Path, bool, str]] = []
    monkeypatch.setattr(
        module,
        "delete_failure_management_folder",
        lambda *, folder_path, cleanup, stage: cleanup_calls.append((folder_path, cleanup, stage)),
    )

    code = module.main()

    assert code == 0
    assert persisted["metrics"] == {"auc": 0.8}
    assert persisted["model_hash"] == "hash123"
    assert persisted["pipeline_hash"] is None
    assert persisted["pipeline_cfg_hash"] is None
    assert persisted["train_run_id"] == "20260306T180000_abcdef01"
    assert cleanup_calls[0][1:] == (True, "train")

    # The following lines of code can be dangerous if the function implementation changes unexpectedly, since they call the actual function. I included it here, since the test leaves empty folders within the actual failure management directory. I use the snippet below to get rid of this harmless, but annoying side-effect of this test. You can manually delete those folders if you want to as well, and the code guarantees that no experiments or training runs will ever have those same IDs, so there's no side effects apart from it being annoying.
    # There are three other tests like this around the repo. Alltogether, they leave folders called "exp_2", "exp_3", "exp_5" and "exp_100" within `failure_management`. This comment should help you understand what is happening and deal with it.
    # I decided to comment the code below out for safety reasons, but leave it in place in case you want to re-enable it for cleanup purposes (I use it locally). Just be mindful of the implications.
    # As of now, I have not yet been able to find time to deal with this issue directly - it's just a nuisance anyway. Of course, you are free to fix it if you find a solution before I do!
    # - Sebastijan

    # leftover_dir = Path("failure_management") / "exp_2" / "training" / "20260306T180000_abcdef01"
    # delete_failure_management_folder(
    #     folder_path=leftover_dir,
    #     cleanup=True,
    #     stage="train"
    # )


def test_main_maps_pipeline_contract_error_when_pipeline_hash_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Convert missing pipeline config hash contract violation into resolved exit code."""
    experiment_dir = tmp_path / "experiments" / "adr" / "global" / "v1" / "exp_3"
    (experiment_dir / "search").mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="adr",
            segment="global",
            version="v1",
            train_run_id=None,
            env="default",
            strict=True,
            experiment_id="exp_3",
            logging_level="INFO",
            clean_up_failure_management=True,
            overwrite_existing=False,
            snapshot_binding_key=None,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T180500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: None)
    monkeypatch.setattr(module, "get_snapshot_path", lambda snapshot_id, parent_dir: experiment_dir)

    cfg = SimpleNamespace(algorithm=SimpleNamespace(value="CatBoost"))
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(module, "add_config_hash", lambda model_cfg: model_cfg)
    monkeypatch.setattr(module, "validate_lineage_integrity", lambda search_dir: None)
    monkeypatch.setattr(module, "validate_reproducibility", lambda runtime_path: None)
    monkeypatch.setattr(module, "validate_logical_config", lambda model_cfg, search_dir: None)
    monkeypatch.setattr(module, "validate_pipeline_cfg", lambda metadata_path, model_cfg: None)

    trainer_output = SimpleNamespace(
        model=object(),
        pipeline=object(),
        pipeline_cfg_hash=None,
        lineage=[],
        metrics={},
    )
    trainer = SimpleNamespace(train=lambda *args, **kwargs: trainer_output)
    monkeypatch.setattr(module, "get_trainer", lambda algorithm: trainer)
    monkeypatch.setattr(module, "save_model", lambda model, train_run_dir: train_run_dir / "model.pkl")
    monkeypatch.setattr(module, "hash_artifact", lambda artifact_path: "hash-model")

    monkeypatch.setattr(
        module,
        "resolve_exit_code",
        lambda e: 88 if isinstance(e, PipelineContractError) else 99,
    )

    code = module.main()

    assert code == 88

    # The following lines of code can be dangerous if the function implementation changes unexpectedly, since they call the actual function. I included it here, since the test leaves empty folders within the actual failure management directory. I use the snippet below to get rid of this harmless, but annoying side-effect of this test. You can manually delete those folders if you want to as well, and the code guarantees that no experiments or training runs will ever have those same IDs, so there's no side effects apart from it being annoying.
    # There are three other tests like this around the repo. Alltogether, they leave folders called "exp_2", "exp_3", "exp_5" and "exp_100" within `failure_management`. This comment should help you understand what is happening and deal with it.
    # I decided to comment the code below out for safety reasons, but leave it in place in case you want to re-enable it for cleanup purposes (I use it locally). Just be mindful of the implications.
    # As of now, I have not yet been able to find time to deal with this issue directly - it's just a nuisance anyway. Of course, you are free to fix it if you find a solution before I do!
    # - Sebastijan

    # leftover_dir = Path("failure_management") / "exp_3" / "training" / "20260306T180500_00112233"
    # delete_failure_management_folder(
    #     folder_path=leftover_dir,
    #     cleanup=True,
    #     stage="train"
    # )


def test_main_returns_one_when_provided_train_run_id_directory_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject user-provided train run IDs that do not map to an existing directory."""
    experiment_dir = tmp_path / "experiments" / "no_show" / "global" / "v1" / "exp_4"
    (experiment_dir / "search").mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            train_run_id="run_missing",
            env="default",
            strict=True,
            experiment_id="exp_4",
            logging_level="INFO",
            clean_up_failure_management=True,
            overwrite_existing=False,
        ),
    )
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(module, "get_snapshot_path", lambda snapshot_id, parent_dir: experiment_dir)

    code = module.main()

    assert code == 1

def test_main_persists_pipeline_artifacts_when_pipeline_and_hash_are_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist pipeline artifact and hashes when trainer returns pipeline + config hash."""
    experiment_dir = tmp_path / "experiments" / "adr" / "global" / "v1" / "exp_5"
    (experiment_dir / "search").mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="adr",
            segment="global",
            version="v1",
            train_run_id=None,
            env="test",
            strict=True,
            experiment_id="exp_5",
            logging_level="INFO",
            clean_up_failure_management=False,
                overwrite_existing=False,
                snapshot_binding_key=None,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260307T130000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="aabbccddeeff0011"))
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: None)
    monkeypatch.setattr(module, "get_snapshot_path", lambda snapshot_id, parent_dir: experiment_dir)

    cfg = SimpleNamespace(algorithm=SimpleNamespace(value="CatBoost"))
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(module, "add_config_hash", lambda model_cfg: model_cfg)
    monkeypatch.setattr(module, "validate_lineage_integrity", lambda search_dir: None)
    monkeypatch.setattr(module, "validate_reproducibility", lambda runtime_path: None)
    monkeypatch.setattr(module, "validate_logical_config", lambda model_cfg, search_dir: None)
    monkeypatch.setattr(module, "validate_pipeline_cfg", lambda metadata_path, model_cfg: None)

    pipeline_obj = object()
    trainer_output = SimpleNamespace(
        model=object(),
        pipeline=pipeline_obj,
        pipeline_cfg_hash="cfg_hash_123",
        lineage=[],
        metrics={"rmse": 1.23},
    )
    trainer = SimpleNamespace(train=lambda *args, **kwargs: trainer_output)
    monkeypatch.setattr(module, "get_trainer", lambda algorithm: trainer)

    model_path = experiment_dir / "training" / "20260307T130000_aabbccdd" / "model.cbm"
    pipeline_path = experiment_dir / "training" / "20260307T130000_aabbccdd" / "pipeline.pkl"
    monkeypatch.setattr(module, "save_model", lambda model, train_run_dir: model_path)
    monkeypatch.setattr(module, "save_pipeline", lambda pipeline, train_run_dir: pipeline_path)

    def _hash_artifact(path: Path) -> str:
        if path == model_path:
            return "model_hash"
        if path == pipeline_path:
            return "pipeline_hash"
        raise AssertionError("Unexpected artifact path")

    monkeypatch.setattr(module, "hash_artifact", _hash_artifact)

    persisted: dict[str, Any] = {}
    monkeypatch.setattr(module, "persist_training_run", lambda model_cfg, **kwargs: persisted.update(kwargs))
    monkeypatch.setattr(module, "delete_failure_management_folder", lambda **kwargs: None)

    code = module.main()

    assert code == 0
    assert persisted["pipeline_path"] == pipeline_path
    assert persisted["pipeline_hash"] == "pipeline_hash"
    assert persisted["pipeline_cfg_hash"] == "cfg_hash_123"

    # The following lines of code can be dangerous if the function implementation changes unexpectedly, since they call the actual function. I included it here, since the test leaves empty folders within the actual failure management directory. I use the snippet below to get rid of this harmless, but annoying side-effect of this test. You can manually delete those folders if you want to as well, and the code guarantees that no experiments or training runs will ever have those same IDs, so there's no side effects apart from it being annoying.
    # There are three other tests like this around the repo. Alltogether, they leave folders called "exp_2", "exp_3", "exp_5" and "exp_100" within `failure_management`. This comment should help you understand what is happening and deal with it.
    # I decided to comment the code below out for safety reasons, but leave it in place in case you want to re-enable it for cleanup purposes (I use it locally). Just be mindful of the implications.
    # As of now, I have not yet been able to find time to deal with this issue directly - it's just a nuisance anyway. Of course, you are free to fix it if you find a solution before I do!
    # - Sebastijan

    # leftover_dir = Path("failure_management") / "exp_5" / "training" / "20260307T130000_aabbccdd"
    # delete_failure_management_folder(
    #     folder_path=leftover_dir,
    #     cleanup=True,
    #     stage="train"
    # )
