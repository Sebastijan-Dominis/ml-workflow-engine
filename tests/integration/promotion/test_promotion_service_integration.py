from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from ml.promotion.context import PromotionContext, PromotionPaths
from ml.promotion.service import PromotionService


def test_promotion_service_runs_and_persists(tmp_path: Path, monkeypatch: Any) -> None:
    # Build a minimal context with tmp paths
    args = SimpleNamespace(
        problem="p",
        segment="s",
        stage="staging",
        version="v1",
        experiment_id="e1",
        train_run_id="t1",
        eval_run_id="e1",
        explain_run_id="x1",
    )

    model_registry_dir = tmp_path / "model_registry"
    paths = PromotionPaths(
        model_registry_dir=model_registry_dir,
        run_dir=model_registry_dir / "runs" / "r1",
        promotion_configs_dir=tmp_path / "configs" / "promotion",
        train_run_dir=tmp_path / "train",
        eval_run_dir=tmp_path / "eval",
        explain_run_dir=tmp_path / "explain",
        registry_path=model_registry_dir / "models.yaml",
        archive_path=model_registry_dir / "archive.yaml",
    )

    # ensure run_dir exists so lock path resolution is stable
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    context = PromotionContext(args=cast(Any, args), run_id="r1", timestamp="ts", paths=paths)

    service = PromotionService()

    # Monkeypatch validators and metadata getter to be no-ops / simple returns
    monkeypatch.setattr("ml.promotion.service.validate_run_dirs", lambda *a, **k: None)
    monkeypatch.setattr(
        "ml.promotion.service.get_runners_metadata",
        lambda *a, **k: SimpleNamespace(explainability_metadata=SimpleNamespace(artifacts=[])),
    )
    monkeypatch.setattr("ml.promotion.service.validate_run_ids", lambda *a, **k: None)
    monkeypatch.setattr("ml.promotion.service.validate_artifacts_consistency", lambda *a, **k: None)
    monkeypatch.setattr("ml.promotion.service.validate_explainability_artifacts", lambda *a, **k: None)

    # Replace state loader and persister on the service instance
    loaded_state = SimpleNamespace(state="st")
    service._state_loader = cast(Any, SimpleNamespace(load=lambda ctx: loaded_state))

    persisted = {}
    def fake_persist(ctx, state, result):
        persisted["ctx"] = ctx
        persisted["state"] = state
        persisted["result"] = result

    service._persister = cast(Any, SimpleNamespace(persist=fake_persist))

    # Provide a fake strategy returned by _get_strategy
    fake_result = {"ok": True}
    class FakeStrategy:
        def execute(self, ctx, state):
            return fake_result

    monkeypatch.setattr(service, "_get_strategy", lambda stage: FakeStrategy())

    # Run and assert
    result = service.run(context)
    assert result == fake_result
    assert persisted["state"] is loaded_state
    assert persisted["result"] == fake_result
