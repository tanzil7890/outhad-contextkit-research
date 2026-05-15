"""Phase A1 — CloudSettings."""
from __future__ import annotations

import pytest


def test_settings_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CLOUD_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    from server.cloud.settings import reset_settings_cache, get_settings

    reset_settings_cache()
    settings = get_settings()
    assert settings.cloud_auth_enabled is False
    reset_settings_cache()


def test_settings_enabled_requires_db_url(monkeypatch):
    monkeypatch.setenv("CLOUD_AUTH_ENABLED", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL_DIRECT", raising=False)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    from server.cloud.settings import reset_settings_cache, get_settings

    reset_settings_cache()
    with pytest.raises(RuntimeError) as excinfo:
        get_settings()
    msg = str(excinfo.value)
    assert "DATABASE_URL" in msg
    assert "CLERK_JWKS_URL" in msg
    reset_settings_cache()


def test_settings_cors_origins_csv(monkeypatch):
    monkeypatch.setenv("CLOUD_AUTH_ENABLED", "false")
    monkeypatch.setenv(
        "CLOUD_CORS_ORIGINS",
        "http://a.com, http://b.com , http://c.com",
    )
    from server.cloud.settings import reset_settings_cache, get_settings

    reset_settings_cache()
    settings = get_settings()
    assert settings.cloud_cors_origins == [
        "http://a.com",
        "http://b.com",
        "http://c.com",
    ]
    reset_settings_cache()
