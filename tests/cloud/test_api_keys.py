"""Phase K1 — generation, hashing, verification."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from server.cloud.auth.api_keys import (
    _hash,
    generate_key,
    issue_api_key,
    revoke_api_key,
    verify_api_key,
)
from server.cloud.auth.clerk import ClerkClaims
from server.cloud.services.users_orgs import ensure_user


@pytest.mark.asyncio
async def test_generate_key_format(cloud_env):
    full, prefix = generate_key(environment="live")
    assert full.startswith("ock_live_")
    assert prefix.startswith("ock_live_")
    assert len(prefix) == len("ock_live_") + 8
    assert _hash(full) == _hash(full)
    full2, _ = generate_key(environment="live")
    assert full != full2


@pytest.mark.asyncio
async def test_generate_key_test_env(cloud_env):
    full, prefix = generate_key(environment="test")
    assert full.startswith("ock_test_")
    assert prefix.startswith("ock_test_")


@pytest.mark.asyncio
async def test_generate_key_invalid_env(cloud_env):
    with pytest.raises(ValueError):
        generate_key(environment="staging")


@pytest.mark.asyncio
async def test_issue_and_verify_round_trip(db_session):
    user, org, project = await ensure_user(
        db_session,
        ClerkClaims(sub="user_a", email="a@x", org_id=None, org_role=None),
    )
    row, full_key = await issue_api_key(
        db_session,
        project_id=project.id,
        created_by_user_id=user.id,
        name="dev",
    )
    await db_session.commit()
    assert row.prefix == full_key.rsplit("_", 1)[0]
    assert row.scopes == ["memory.read", "memory.write"]

    ctx = await verify_api_key(db_session, full_key)
    assert ctx.user_id == user.id
    assert ctx.organization_id == org.id
    assert ctx.project_id == project.id
    assert ctx.tenant_id == org.tenant_id
    assert ctx.role == "OWNER"
    assert ctx.api_key_id == row.id
    assert ctx.auth_method == "api_key"


@pytest.mark.asyncio
async def test_verify_unknown_key_raises_401(db_session):
    with pytest.raises(HTTPException) as excinfo:
        await verify_api_key(db_session, "ock_live_xxxxxxxx_unknown")
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_wrong_prefix_rejected(db_session):
    with pytest.raises(HTTPException) as excinfo:
        await verify_api_key(db_session, "sk_live_should_not_pass")
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_revoke_then_verify_fails(db_session):
    user, org, project = await ensure_user(
        db_session,
        ClerkClaims(sub="user_b", email="b@x", org_id=None, org_role=None),
    )
    row, full_key = await issue_api_key(
        db_session,
        project_id=project.id,
        created_by_user_id=user.id,
        name="dev",
    )
    await db_session.commit()

    ok = await revoke_api_key(
        db_session, api_key_id=row.id, organization_id=org.id
    )
    await db_session.commit()
    assert ok is True

    with pytest.raises(HTTPException) as excinfo:
        await verify_api_key(db_session, full_key)
    assert excinfo.value.status_code == 401
    assert "revoked" in str(excinfo.value.detail).lower()


@pytest.mark.asyncio
async def test_revoke_cross_tenant_blocked(db_session):
    user_a, org_a, proj_a = await ensure_user(
        db_session,
        ClerkClaims(sub="user_a", email="a@x", org_id=None, org_role=None),
    )
    row_a, _ = await issue_api_key(
        db_session,
        project_id=proj_a.id,
        created_by_user_id=user_a.id,
        name="a-key",
    )

    user_b, org_b, proj_b = await ensure_user(
        db_session,
        ClerkClaims(sub="user_b", email="b@x", org_id=None, org_role=None),
    )
    await db_session.commit()

    ok = await revoke_api_key(
        db_session, api_key_id=row_a.id, organization_id=org_b.id
    )
    assert ok is False
