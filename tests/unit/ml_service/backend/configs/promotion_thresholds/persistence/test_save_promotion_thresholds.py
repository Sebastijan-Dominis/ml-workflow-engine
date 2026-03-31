import importlib


def test_save_with_existing_thresholds(monkeypatch, tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.promotion_thresholds.persistence.save_promotion_thresholds"
    )

    saved = {}

    def fake_save(cfg, pth):
        saved["cfg"] = cfg
        saved["pth"] = pth

    monkeypatch.setattr(mod, "save_config", fake_save)

    class FakeValidated:
        def model_dump(self, mode="json"):
            return {"new": True}

    thresholds = {"regression": {"s0": {"threshold": 0.1}}}
    validated = FakeValidated()
    config_path = tmp_path / "pth.yaml"

    mod.save_promotion_thresholds(
        thresholds=thresholds,
        validated=validated,
        config_path=config_path,
        problem_type="regression",
        segment="s1",
    )

    assert "regression" in saved["cfg"]
    assert "s0" in saved["cfg"]["regression"]
    assert saved["cfg"]["regression"]["s1"] == {"new": True}
    assert saved["pth"] == config_path


def test_save_with_none_thresholds(monkeypatch, tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.promotion_thresholds.persistence.save_promotion_thresholds"
    )

    recorded = {}

    def fake_save(cfg, pth):
        recorded["cfg"] = cfg
        recorded["pth"] = pth

    monkeypatch.setattr(mod, "save_config", fake_save)

    class FakeValidated:
        def model_dump(self, mode="json"):
            return {"v": 1}

    mod.save_promotion_thresholds(
        thresholds=None,
        validated=FakeValidated(),
        config_path=tmp_path / "out.yaml",
        problem_type="classification",
        segment="sA",
    )

    assert recorded["cfg"]["classification"]["sA"] == {"v": 1}


def test_save_propagates_error(monkeypatch, tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.promotion_thresholds.persistence.save_promotion_thresholds"
    )

    def bad_save(cfg, pth):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mod, "save_config", bad_save)

    class FakeValidated:
        def model_dump(self, mode="json"):
            return {"x": 2}

    try:
        mod.save_promotion_thresholds(
            thresholds={},
            validated=FakeValidated(),
            config_path=tmp_path / "out.yaml",
            problem_type="p",
            segment="s",
        )
        raised = False
    except RuntimeError as e:
        raised = True
        assert "disk full" in str(e)

    assert raised
