import importlib
from types import SimpleNamespace


def test_pipelines_infer_main(monkeypatch):
    mod = importlib.import_module("pipelines.post_promotion.infer")

    args = SimpleNamespace(problem="p", segment="s", snapshot_bindings_id="sb", logging_level="INFO")
    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(mod, "setup_logging", lambda *a, **kw: None)

    from ml.promotion.config.registry_entry import (
        RegistryArtifacts,
        RegistryEntry,
        RegistryEntryMetrics,
    )

    reg = RegistryEntry(
        experiment_id="e",
        train_run_id="t",
        eval_run_id="x",
        explain_run_id="r",
        model_version="mv",
        artifacts=RegistryArtifacts(model_hash="mh", model_path="mp"),
        feature_lineage=[],
        metrics=RegistryEntryMetrics(train={}, val={}, test={}),
        git_commit="c",
    )

    monkeypatch.setattr(mod, "get_model_registry_info", lambda a: SimpleNamespace(prod_meta=reg, stage_meta=None))

    calls = []
    monkeypatch.setattr(mod, "execute_inference", lambda **kwargs: calls.append(kwargs))

    rc = mod.main()

    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["stage"] == "production"
