"""Phase A9 — ensure_user idempotency."""
from __future__ import annotations

import pytest
from sqlalchemy import select

from server.cloud.auth.clerk import ClerkClaims
from server.cloud.models import AuditLog, Member, Organization, Project, User
from server.cloud.services.users_orgs import ensure_user


def _claims(sub: str = "user_abc") -> ClerkClaims:
    return ClerkClaims(sub=sub, email="x@example.com", org_id=None, org_role=None)


@pytest.mark.asyncio
async def test_first_signup_creates_full_triple(db_session):
    user, org, project = await ensure_user(db_session, _claims())
    await db_session.commit()

    assert user.email == "x@example.com"
    assert org.tenant_id.startswith("org_")
    assert org.owner_user_id == user.id
    assert project.organization_id == org.id
    assert project.slug == "default"

    member = (
        await db_session.execute(
            select(Member).where(Member.user_id == user.id)
        )
    ).scalar_one()
    assert member.role == "OWNER"

    audit = (
        await db_session.execute(
            select(AuditLog).where(AuditLog.user_id == user.id)
        )
    ).scalars().all()
    assert any(row.action == "user.signup" for row in audit)


@pytest.mark.asyncio
async def test_repeat_signin_is_idempotent(db_session):
    user1, org1, project1 = await ensure_user(db_session, _claims())
    await db_session.commit()
    user2, org2, project2 = await ensure_user(db_session, _claims())
    await db_session.commit()

    assert user1.id == user2.id
    assert org1.id == org2.id
    assert project1.id == project2.id

    user_count = (
        await db_session.execute(select(User))
    ).scalars().all()
    assert len(user_count) == 1
