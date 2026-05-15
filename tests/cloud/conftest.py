"""Shared fixtures for cloud auth tests.

Every fixture works against in-memory SQLite via ``aiosqlite`` so the
unit tests run in milliseconds offline. The schema is built directly
from ``Base.metadata`` (the alembic migration is asserted separately
in ``test_models.py``).
"""
from __future__ import annotations

import os
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


_TEST_WEBHOOK_SECRET = "whsec_" + ("A" * 40)


def _set_cloud_env(monkeypatch_obj) -> None:
    monkeypatch_obj.setenv("CLOUD_AUTH_ENABLED", "true")
    monkeypatch_obj.setenv(
        "DATABASE_URL", "sqlite+aiosqlite:///:memory:"
    )
    monkeypatch_obj.setenv("CLERK_PUBLISHABLE_KEY", "pk_test_x")
    monkeypatch_obj.setenv("CLERK_SECRET_KEY", "sk_test_x")
    monkeypatch_obj.setenv(
        "CLERK_JWKS_URL", "https://example.clerk.accounts.dev/.well-known/jwks.json"
    )
    monkeypatch_obj.setenv("CLERK_ISSUER", "https://example.clerk.accounts.dev")
    monkeypatch_obj.setenv("CLERK_WEBHOOK_SECRET", _TEST_WEBHOOK_SECRET)


@pytest.fixture
def cloud_env(monkeypatch):
    _set_cloud_env(monkeypatch)
    from server.cloud.settings import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest_asyncio.fixture
async def async_engine(cloud_env) -> AsyncIterator[AsyncEngine]:
    from server.cloud.db import configure_engine, dispose_engine
    from server.cloud.models import Base

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", future=True
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    configure_engine(engine)
    try:
        yield engine
    finally:
        await dispose_engine()
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    sm = async_sessionmaker(bind=async_engine, expire_on_commit=False)
    async with sm() as session:
        yield session
        await session.rollback()
