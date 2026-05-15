import os
from unittest.mock import patch

import pytest

OUTHAD_CONTEXTKIT_TELEMETRY = os.environ.get("OUTHAD_CONTEXTKIT_TELEMETRY", "True")

if isinstance(OUTHAD_CONTEXTKIT_TELEMETRY, str):
    OUTHAD_CONTEXTKIT_TELEMETRY = OUTHAD_CONTEXTKIT_TELEMETRY.lower() in ("true", "1", "yes")


def use_telemetry():
    if os.getenv("OUTHAD_CONTEXTKIT_TELEMETRY", "true").lower() == "true":
        return True
    return False


@pytest.fixture(autouse=True)
def reset_env():
    with patch.dict(os.environ, {}, clear=True):
        yield


def test_telemetry_enabled():
    with patch.dict(os.environ, {"OUTHAD_CONTEXTKIT_TELEMETRY": "true"}):
        assert use_telemetry() is True


def test_telemetry_disabled():
    with patch.dict(os.environ, {"OUTHAD_CONTEXTKIT_TELEMETRY": "false"}):
        assert use_telemetry() is False


def test_telemetry_default_enabled():
    assert use_telemetry() is True
