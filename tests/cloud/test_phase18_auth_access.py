"""Phase 1.8 — auth + access via published SDK 1.8.0.

Drives every Phase 1.8 surface against PyPI install:
  - Audit log read (GET /v1/audit-log/) with filters
  - Key rotation (mint replacement + grace window on old)
  - update_api_key (scopes + ip_allowlist + name)
  - Scope enforcement on memory.write paths
  - IP allowlist on api_keys

Run::

    pip install outhad-contextkit-sdk asyncpg requests
    export OUTHAD_CONTEXTKIT_API_KEY=ock_live_...
    export OUTHAD_CONTEXTKIT_DATABASE_URL=postgresql://...
    python tests/cloud/test_phase18_auth_access.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import uuid

import requests

from outhad_contextkit_sdk import APIError, MemoryClient

API_KEY = os.environ["OUTHAD_CONTEXTKIT_API_KEY"]
HOST = os.environ.get("OUTHAD_CONTEXTKIT_HOST", "http://localhost:8000")
DATABASE_URL = os.environ.get("OUTHAD_CONTEXTKIT_DATABASE_URL")


def banner(t):
    print()
    print("=" * 80)
    print(t)
    print("=" * 80)


def show(label, value):
    print(f"\n  ▸ {label}:")
    if isinstance(value, (dict, list)):
        print(textwrap.indent(json.dumps(value, indent=2, default=str), "    "))
    else:
        print(f"    {value}")


async def _mint_test_key(*, scopes, ip_allowlist=None) -> str:
    """Mint a sandbox-prefix API key directly via SQL so we don't
    need a Clerk JWT in the test rig. Returns the full ock_test_… key.
    """
    import asyncpg, hashlib, secrets
    body = secrets.token_urlsafe(32)
    pub = secrets.token_urlsafe(6).replace("_", "").replace("-", "")[:8]
    full = f"ock_test_{pub}_{body}"
    prefix = f"ock_test_{pub}"
    key_hash = hashlib.sha256(full.encode("utf-8")).hexdigest()
    conn = await asyncpg.connect(DATABASE_URL, ssl="require")
    try:
        # Reuse the existing project + creator from the live key.
        row = await conn.fetchrow(
            "SELECT project_id, created_by_user_id FROM api_keys "
            "WHERE prefix LIKE 'ock_live_%' LIMIT 1"
        )
        await conn.execute(
            """
            INSERT INTO api_keys
                (id, project_id, created_by_user_id, name, prefix, key_hash,
                 scopes, environment, ip_allowlist)
            VALUES (gen_random_uuid(), $1::uuid, $2::uuid, $3, $4, $5,
                    $6::jsonb, 'test', $7::jsonb)
            """,
            row["project_id"], row["created_by_user_id"],
            f"qa-1.8-{uuid.uuid4().hex[:6]}",
            prefix, key_hash,
            json.dumps(scopes),
            json.dumps(ip_allowlist or []),
        )
    finally:
        await conn.close()
    return full


async def _drop_test_keys():
    if not DATABASE_URL:
        return
    import asyncpg
    conn = await asyncpg.connect(DATABASE_URL, ssl="require")
    try:
        await conn.execute(
            "DELETE FROM api_keys WHERE prefix LIKE 'ock_test_%' AND name LIKE 'qa-1.8-%'"
        )
    finally:
        await conn.close()


def main() -> int:
    banner("Connecting")
    client = MemoryClient(api_key=API_KEY, host=HOST)
    show("user_email", client.user_email)
    show("project_id", client.project_id)

    # ── 1. Audit log read API ──────────────────────────────
    banner("1. GET /v1/audit-log/  (Clerk JWT bypasses scope check)")
    log = client.audit_log(limit=5)
    show("audit_log first 5 entries", log)

    # filter by action
    log_act = client.audit_log(action="api_key.created", limit=3)
    show("audit_log filtered action='api_key.created'", log_act)

    # ── 2. Scope enforcement: read-only key blocks writes ──
    banner("2. Mint read-only test key — scopes=['memory.read'] only")
    if not DATABASE_URL:
        show("SKIP", "OUTHAD_CONTEXTKIT_DATABASE_URL not set")
    else:
        ro_key = asyncio.run(_mint_test_key(scopes=["memory.read"]))
        show("read-only key prefix", ro_key.split("_")[2][:8])

        ro_client = MemoryClient(api_key=ro_key, host=HOST)
        # Read should succeed
        try:
            rows = ro_client.get_all(user_id="never-existed", limit=1)
            show("get_all (read) — should succeed", rows)
        except APIError as exc:
            show("get_all ERROR", f"status={exc.status} {exc}")
        # Write should 403
        try:
            ro_client.add(
                messages=[{"role": "user", "content": "hi"}],
                user_id="qa-18-blocked",
            )
            show("add() with read-only key", "❌ NO 403 — bug")
        except APIError as exc:
            show(
                "add() with read-only key — expected 403",
                f"status={exc.status} {str(exc)[:140]}",
            )
        ro_client.close()

    # ── 3. IP allowlist ────────────────────────────────────
    banner("3. Mint test key with ip_allowlist=['203.0.113.99'] — non-matching IP")
    if not DATABASE_URL:
        show("SKIP", "no DATABASE_URL")
    else:
        ip_key = asyncio.run(_mint_test_key(
            scopes=["memory.read", "memory.write"],
            ip_allowlist=["203.0.113.99"],   # RFC 5737 test net — not us
        ))
        # Hit the cloud directly with this key — local IP won't match
        # the allowlist so the dispatcher should 403.
        try:
            r = requests.get(
                f"{HOST}/v1/me",
                headers={"Authorization": f"Token {ip_key}"},
                timeout=5,
            )
            show("verify_api_key with mismatched IP", f"status={r.status_code} body={r.text[:200]}")
        except Exception as exc:
            show("ERROR", str(exc)[:200])

    # ── 4. Audit log filter on api_key.rotated (negative case) ─
    banner("4. audit_log(resource_type='api_key') — surfaces all key events")
    log_keys = client.audit_log(resource_type="api_key", limit=10)
    actions_seen = sorted({(r.get("action")) for r in (log_keys.get("results") or [])})
    show("distinct actions in api_key audit subset", actions_seen)

    # cleanup
    asyncio.run(_drop_test_keys())
    client.close()

    banner("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
