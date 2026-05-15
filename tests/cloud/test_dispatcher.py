"""Phase A8 — auth dispatcher routing."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from server.cloud.auth.api_keys import issue_api_key
from server.cloud.auth.clerk import ClerkClaims
from server.cloud.auth.dispatcher import _extract_token, get_auth_context
from server.cloud.services.users_orgs import ensure_user


def _request(headers: dict) -> SimpleNamespace:
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host="127.0.0.1"),
    )


def test_extract_x_api_key():
    req = _request({"x-api-key": "ock_live_xxxxxxxx_abc"})
    assert _extract_token(req) == ("api_key", "ock_live_xxxxxxxx_abc")


def test_extract_bearer_api_key():
    req = _request({"authorization": "Bearer ock_test_xxxxxxxx_abc"})
    assert _extract_token(req) == ("api_key", "ock_test_xxxxxxxx_abc")


def test_extract_bearer_jwt():
    req = _request({"authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.x.y"})
    assert _extract_token(req) == ("clerk_jwt", "eyJhbGciOiJSUzI1NiJ9.x.y")


def test_extract_no_credentials():
    req = _request({})
    assert _extract_token(req) is None


def test_extract_malformed_authorization():
    req = _request({"authorization": "Basic abc"})
    assert _extract_token(req) is None


@pytest.mark.asyncio
async def test_dispatcher_resolves_api_key_path(db_session):
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

    request = _request({"x-api-key": full_key})
    ctx = await get_auth_context(request, db=db_session)
    assert ctx.auth_method == "api_key"
    assert ctx.api_key_id == row.id


@pytest.mark.asyncio
async def test_dispatcher_401_when_missing(db_session):
    request = _request({})
    with pytest.raises(HTTPException) as excinfo:
        await get_auth_context(request, db=db_session)
    assert excinfo.value.status_code == 401
