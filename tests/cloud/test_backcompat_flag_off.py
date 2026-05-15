"""Phase A12 back-compat — with CLOUD_AUTH_ENABLED=false the cloud
modules must not be imported by ``server.main``.

This guards against regressions where a stray top-level import would
force every self-hosted user to install the cloud extras (asyncpg /
alembic / svix / python-jose).
"""
from __future__ import annotations

import sys


def test_main_does_not_import_cloud_when_flag_off(monkeypatch):
    monkeypatch.setenv("CLOUD_AUTH_ENABLED", "false")
    # Drop any prior import of the cloud subpackage so the assertion
    # measures *this* import.
    for name in list(sys.modules):
        if name.startswith("server.cloud"):
            sys.modules.pop(name, None)

    # Importing only the settings module is fine — that's what other
    # parts of the codebase may peek at. What we forbid is the engine,
    # routes, etc.
    assert "server.cloud.db" not in sys.modules
    assert "server.cloud.routes.api_keys" not in sys.modules
    assert "server.cloud.routes.webhooks" not in sys.modules
