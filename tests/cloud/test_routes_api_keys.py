"""Phase K2 — /v1/api-keys REST endpoints."""
from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.cloud.auth.api_keys import issue_api_key
from server.cloud.auth.clerk import ClerkClaims
from server.cloud.auth.context import AuthContext
from server.cloud.auth.dispatcher import get_auth_context
from server.cloud.db import get_db
from server.cloud.routes.api_keys import router
from server.cloud.services.users_orgs import ensure_user


def _build_app(db_session, auth_factory):
    app = FastAPI()
    app.include_router(router)

    async def override_db():
        yield db_session

    async def override_auth():
        return await auth_factory()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_auth_context] = override_auth
    return app


def _ctx_from(user, org, project, *, method: str = "clerk_jwt") -> AuthContext:
    return AuthContext(
        user_id=user.id,
        organization_id=org.id,
        project_id=project.id,
        tenant_id=org.tenant_id,
        sub_tenant_id=project.sub_tenant_id,
        role="OWNER",
        api_key_id=None,
        auth_method=method,
    )


@pytest.mark.asyncio
async def test_create_then_list(db_session):
    user, org, project = await ensure_user(
        db_session,
        ClerkClaims(sub="user_1", email="u@x", org_id=None, org_role=None),
    )
    await db_session.commit()

    async def auth_factory():
        return _ctx_from(user, org, project)

    app = _build_app(db_session, auth_factory)
    with TestClient(app) as client:
        create = client.post("/v1/api-keys", json={"name": "dev"})
        assert create.status_code == 201, create.text
        body = create.json()
        assert body["full_key"].startswith("ock_live_")
        assert body["prefix"].startswith("ock_live_")

        listing = client.get("/v1/api-keys")
        assert listing.status_code == 200
        rows = listing.json()
        assert len(rows) == 1
        assert "full_key" not in rows[0]


@pytest.mark.asyncio
async def test_api_key_caller_cannot_create_more(db_session):
    user, org, project = await ensure_user(
        db_session,
        ClerkClaims(sub="user_2", email="u@x", org_id=None, org_role=None),
    )
    await db_session.commit()

    async def auth_factory():
        return _ctx_from(user, org, project, method="api_key")

    app = _build_app(db_session, auth_factory)
    with TestClient(app) as client:
        resp = client.post("/v1/api-keys", json={"name": "loop"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_revoke_removes_key(db_session):
    user, org, project = await ensure_user(
        db_session,
        ClerkClaims(sub="user_3", email="u@x", org_id=None, org_role=None),
    )
    row, _ = await issue_api_key(
        db_session,
        project_id=project.id,
        created_by_user_id=user.id,
        name="dev",
    )
    await db_session.commit()

    async def auth_factory():
        return _ctx_from(user, org, project)

    app = _build_app(db_session, auth_factory)
    with TestClient(app) as client:
        delete = client.delete(f"/v1/api-keys/{row.id}")
        assert delete.status_code == 204

        again = client.delete(f"/v1/api-keys/{row.id}")
        assert again.status_code == 404


@pytest.mark.asyncio
async def test_revoke_invalid_uuid_400(db_session):
    user, org, project = await ensure_user(
        db_session,
        ClerkClaims(sub="user_4", email="u@x", org_id=None, org_role=None),
    )
    await db_session.commit()

    async def auth_factory():
        return _ctx_from(user, org, project)

    app = _build_app(db_session, auth_factory)
    with TestClient(app) as client:
        resp = client.delete("/v1/api-keys/not-a-uuid")
    assert resp.status_code == 400
