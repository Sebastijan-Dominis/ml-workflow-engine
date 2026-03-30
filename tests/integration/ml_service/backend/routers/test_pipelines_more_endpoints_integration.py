"""Integration tests exercising additional `pipelines` router endpoints.

These tests stub the underlying `execute_pipeline` helper and assert the
expected module paths are invoked for each endpoint.
"""

from typing import Any

import ml_service.backend.routers.pipelines as pipelines_router


def test_pipelines_various_endpoints(monkeypatch: Any, fastapi_client: Any) -> None:
    called: dict[str, Any] = {"calls": []}

    def fake_execute_pipeline(module_path: str, payload, boolean_args=None):
        called["calls"].append({"module_path": module_path, "payload": getattr(payload, "model_dump", lambda **k: dict(payload))()})
        return {"exit_code": 0, "status": "SUCCESS", "stdout": "", "stderr": ""}

    monkeypatch.setattr(pipelines_router, "execute_pipeline", fake_execute_pipeline)

    endpoint_to_module = {
        "/pipelines/register_raw_snapshot": "pipelines.data.register_raw_snapshot",
        "/pipelines/build_interim_dataset": "pipelines.data.build_interim_dataset",
        "/pipelines/build_processed_dataset": "pipelines.data.build_processed_dataset",
        "/pipelines/freeze_feature_set": "pipelines.features.freeze",
        "/pipelines/evaluate": "pipelines.runners.evaluate",
        "/pipelines/explain": "pipelines.runners.explain",
        "/pipelines/promote": "pipelines.promotion.promote",
        "/pipelines/execute_all_data_preprocessing": "pipelines.orchestration.data.execute_all_data_preprocessing",
        "/pipelines/freeze_all_feature_sets": "pipelines.orchestration.features.freeze_all_feature_sets",
        "/pipelines/execute_experiment_with_latest": "pipelines.orchestration.experiments.execute_experiment_with_latest",
        "/pipelines/execute_all_experiments_with_latest": "pipelines.orchestration.experiments.execute_all_experiments_with_latest",
        "/pipelines/infer": "pipelines.post_promotion.infer",
        "/pipelines/monitor": "pipelines.post_promotion.monitor",
    }

    payloads: dict[str, dict[str, Any]] = {
        "/pipelines/register_raw_snapshot": {"data": "hotel_bookings", "version": "v1"},
        "/pipelines/build_interim_dataset": {"data": "hotel_bookings", "version": "v1"},
        "/pipelines/build_processed_dataset": {"data": "hotel_bookings", "version": "v1"},
        "/pipelines/freeze_feature_set": {"feature_set": "feat", "version": "v1"},
        "/pipelines/evaluate": {"problem": "cancellation", "segment": "all", "version": "v1"},
        "/pipelines/explain": {"problem": "cancellation", "segment": "all", "version": "v1"},
        "/pipelines/promote": {"problem": "cancellation", "segment": "all", "version": "v1", "experiment_id": "e", "train_run_id": "t", "eval_run_id": "ev", "explain_run_id": "ex", "stage": "production"},
        "/pipelines/execute_all_data_preprocessing": {},
        "/pipelines/freeze_all_feature_sets": {},
        "/pipelines/execute_experiment_with_latest": {"problem": "cancellation", "segment": "all", "version": "v1"},
        "/pipelines/execute_all_experiments_with_latest": {},
        "/pipelines/infer": {"problem": "cancellation", "segment": "all", "snapshot_bindings_id": "id"},
        "/pipelines/monitor": {"problem": "cancellation", "segment": "all"},
    }

    for endpoint, expected_module in endpoint_to_module.items():
        resp = fastapi_client.post(endpoint, json=payloads.get(endpoint, {}))
        assert resp.status_code == 200
        body = resp.json()
        assert body["exit_code"] == 0
        # find a call matching expected module
        assert any(c.get("module_path") == expected_module for c in called["calls"]) is True
