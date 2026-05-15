"""Phase A3 — ORM models round-trip on SQLite."""
from __future__ import annotations

import uuid

import pytest

from server.cloud.models import (
    ApiKey,
    AuditLog,
    Member,
    Organization,
    Project,
    UsageMeter,
    User,
)


@pytest.mark.asyncio
async def test_full_lifecycle(db_session):
    user = User(clerk_user_id="user_x", email="x@example.com")
    db_session.add(user)
    await db_session.flush()

    org = Organization(
        name="x's org",
        slug="org-x",
        tenant_id=f"org_{user.id.hex}",
        owner_user_id=user.id,
    )
    db_session.add(org)
    await db_session.flush()

    project = Project(
        organization_id=org.id,
        name="default",
        slug="default",
        sub_tenant_id=f"sub_{org.id.hex}",
    )
    db_session.add(project)
    await db_session.flush()

    member = Member(
        organization_id=org.id, user_id=user.id, role="OWNER"
    )
    db_session.add(member)
    await db_session.flush()

    key = ApiKey(
        project_id=project.id,
        created_by_user_id=user.id,
        name="dev-key",
        prefix="ock_test_aaaaaaaa",
        key_hash="0" * 64,
        scopes=["memory.read"],
    )
    db_session.add(key)
    await db_session.flush()

    audit = AuditLog(
        action="api_key.created",
        user_id=user.id,
        organization_id=org.id,
        api_key_id=key.id,
        metadata_json={"name": "dev-key"},
    )
    db_session.add(audit)

    meter = UsageMeter(
        project_id=project.id,
        api_key_id=key.id,
        bucket="memory.add",
        period="2026-04-25",
        count=1,
    )
    db_session.add(meter)
    await db_session.flush()

    assert isinstance(user.id, uuid.UUID)
    assert org.tenant_id.startswith("org_")
    assert key.scopes == ["memory.read"]
