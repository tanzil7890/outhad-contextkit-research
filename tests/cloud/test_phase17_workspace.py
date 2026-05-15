"""Phase 1.7 — env routing / multilingual / memory_depth / usecase_setting /
project-scoped api keys via published SDK 1.7.0.

Run::

    pip install outhad-contextkit-sdk asyncpg
    export OUTHAD_CONTEXTKIT_API_KEY=ock_live_...
    export OUTHAD_CONTEXTKIT_DATABASE_URL=postgresql://...
    python tests/cloud/test_phase17_workspace.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import uuid

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


async def _set_field(project_id, field, value):
    if not DATABASE_URL:
        raise RuntimeError("OUTHAD_CONTEXTKIT_DATABASE_URL not set")
    import asyncpg
    conn = await asyncpg.connect(DATABASE_URL, ssl="require")
    try:
        await conn.execute(
            f"UPDATE projects SET {field}=$1 WHERE id=$2::uuid", value, project_id
        )
    finally:
        await conn.close()


def _peek_collections(project_hex):
    """Pgvector collections live on the local docker-compose Postgres
    container (not Neon — Neon holds the cloud control plane).
    Inspect via ``docker exec`` so the test doesn't need a separate
    connection string."""
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "docker", "exec", "outhad-cloud-dev-postgres-1",
                "psql", "-U", "postgres", "-d", "postgres",
                "-A", "-t", "-c",
                f"SELECT tablename FROM pg_tables WHERE tablename LIKE 'mem_{project_hex}_%';",
            ],
            stderr=subprocess.DEVNULL, text=True,
        )
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def main() -> int:
    banner("Connecting")
    client = MemoryClient(api_key=API_KEY, host=HOST)
    show("user_email", client.user_email)
    show("project_id", client.project_id)

    if not DATABASE_URL:
        print("\n  OUTHAD_CONTEXTKIT_DATABASE_URL not set — exiting")
        return 0

    project_id = client.project_id
    project_hex = uuid.UUID(project_id).hex[:8]

    try:
        # ── 1. Project surfaces new fields ─────────────────
        banner("1. GET /v1/projects/{id} surfaces multilingual / memory_depth / usecase_setting")
        proj = client.get_project(project_id)
        show("project (relevant fields)", {
            "multilingual": proj.get("multilingual"),
            "memory_depth": proj.get("memory_depth"),
            "usecase_setting": proj.get("usecase_setting"),
        })

        # ── 2. Toggle them via direct SQL (Clerk would do PATCH) ─
        banner("2. Set multilingual=True + memory_depth=20 + usecase_setting='support'")
        asyncio.run(_set_field(project_id, "multilingual", True))
        asyncio.run(_set_field(project_id, "memory_depth", 20))
        asyncio.run(_set_field(project_id, "usecase_setting", "support"))
        proj2 = client.get_project(project_id)
        show("project after SQL update", {
            "multilingual": proj2.get("multilingual"),
            "memory_depth": proj2.get("memory_depth"),
            "usecase_setting": proj2.get("usecase_setting"),
        })

        # ── 3. Add a memory and confirm fact extractor still works ──
        banner("3. add() should still work with new directives baked into the prompt")
        uid = f"qa-17-{uuid.uuid4().hex[:6]}"
        out = client.add(
            messages=[{"role": "user", "content": "I'm having trouble logging in to the dashboard."}],
            user_id=uid,
        )
        show("add response", out)

        # ── 4. List existing pgvector tables — env suffix should appear ──
        banner("4. pgvector tables under this project — should show env in the name")
        tables = _peek_collections(project_hex)
        show("collections", tables)
        live_tables = [t for t in tables if "_live_" in t]
        show("live tables (should be > 0)", len(live_tables))

        # ── 5. Reset workspace knobs ──
        asyncio.run(_set_field(project_id, "multilingual", False))
        asyncio.run(_set_field(project_id, "memory_depth", 10))
        asyncio.run(_set_field(project_id, "usecase_setting", None))

        # ── 6. cleanup user ──
        try:
            client.delete_users(user_id=uid)
        except Exception:
            pass
    finally:
        client.close()

    banner("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
