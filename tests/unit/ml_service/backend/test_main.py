import asyncio
import importlib


def test_health_check_async():
    mod = importlib.import_module("ml_service.backend.main")
    res = asyncio.get_event_loop().run_until_complete(mod.health_check())
    assert res == {"Healthy": 200}


def test_rate_limit_exceeded_handler_returns_429():
    mod = importlib.import_module("ml_service.backend.main")
    # Call the async handler synchronously
    # The handler does not inspect the exception, so a plain Exception is fine
    resp = asyncio.get_event_loop().run_until_complete(
        mod.rate_limit_exceeded_handler(None, Exception("rlimit"))
    )
    assert getattr(resp, "status_code", None) == 429
    body = getattr(resp, "body", None)
    assert body is not None
    assert b"Rate limit exceeded" in body
