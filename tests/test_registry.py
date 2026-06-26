import pytest

from zamba.models import registry

pytestmark = pytest.mark.video


def test_ensure_registered_loads_video_models():
    registry._registered = False
    registry.ensure_registered()
    assert registry._registered
    assert "TimeDistributedEfficientNet" in registry.available_models
    assert "SlowFast" in registry.available_models


def test_ensure_registered_is_idempotent():
    registry.ensure_registered()
    models_after_first = dict(registry.available_models)
    registry.ensure_registered()
    assert registry.available_models == models_after_first
