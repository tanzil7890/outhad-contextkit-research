"""Phase A10 — Clerk inbound webhooks."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select
from svix.webhooks import Webhook

from server.cloud.auth.api_keys import issue_api_key
from server.cloud.auth.clerk import ClerkClaims
from server.cloud.db import get_db
from server.cloud.models import ApiKey, User
from server.cloud.routes.webhooks import router as webhooks_router
from server.cloud.services.users_orgs import ensure_user


import os


def _webhook_secret() -> str:
    return os.environ["CLERK_WEBHOOK_SECRET"]


def _build_app(db_session):
    app = FastAPI()
    app.include_router(webhooks_router)

    async def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    return app


def _signed_headers(payload: bytes, msg_id: str = "msg_1") -> dict:
    import datetime as _dt

    wh = Webhook(_webhook_secret())
    timestamp = _dt.datetime.now(tz=_dt.timezone.utc)
    signature = wh.sign(msg_id, timestamp, payload.decode("utf-8"))
    return {
        "svix-id": msg_id,
        "svix-timestamp": str(int(timestamp.timestamp())),
        "svix-signature": signature,
    }


def _ts() -> str:
    import time

    return str(int(time.time()))


@pytest.mark.asyncio
async def test_user_deleted_revokes_keys(db_session, cloud_env):
    user, org, project = await ensure_user(
        db_session,
        ClerkClaims(sub="user_to_delete", email="d@x", org_id=None, org_role=None),
    )
    row, _ = await issue_api_key(
        db_session,
        project_id=project.id,
        created_by_user_id=user.id,
        name="will-be-revoked",
    )
    await db_session.commit()

    payload = json.dumps(
        {"type": "user.deleted", "data": {"id": "user_to_delete"}}
    ).encode("utf-8")
    headers = _signed_headers(payload)

    app = _build_app(db_session)
    with TestClient(app) as client:
        resp = client.post("/webhooks/clerk", content=payload, headers=headers)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["event"] == "user.deleted"
    assert body["deleted"] == 1
    assert body["revoked_api_keys"] == 1

    refreshed_key = (
        await db_session.execute(select(ApiKey).where(ApiKey.id == row.id))
    ).scalar_one()
    assert refreshed_key.revoked_at is not None
    refreshed_user = (
        await db_session.execute(select(User).where(User.id == user.id))
    ).scalar_one()
    assert refreshed_user.is_active is False


def test_bad_signature_rejected(db_session, cloud_env):
    payload = json.dumps({"type": "user.deleted", "data": {"id": "x"}}).encode()
    bogus_headers = {
        "svix-id": "msg_x",
        "svix-timestamp": _ts(),
        "svix-signature": "v1,not-a-real-signature",
    }

    app = _build_app(db_session)
    with TestClient(app) as client:
        resp = client.post(
            "/webhooks/clerk", content=payload, headers=bogus_headers
        )
    assert resp.status_code == 400
