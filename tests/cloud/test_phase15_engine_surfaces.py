"""Phase 1.5 — every engine surface end-to-end via published SDK 1.5.0.

Mirrors examples/phase15/local_engine_surfaces.py — same 4 memories,
same updates, same feedback / reference / decay / archive_low /
cold-storage / backfill / reset flow — but driven entirely from the
public SDK against the cloud.

Run::

    pip install outhad-contextkit-sdk asyncpg
    export OUTHAD_CONTEXTKIT_API_KEY=ock_live_...
    export OUTHAD_CONTEXTKIT_DATABASE_URL=postgresql://...
    python tests/cloud/test_phase15_engine_surfaces.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
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


def main() -> int:
    banner("Connecting")
    client = MemoryClient(api_key=API_KEY, host=HOST)
    show("user_email", client.user_email)
    show("project_id", client.project_id)
    show("host", client.host)

    if not DATABASE_URL:
        print("\n  OUTHAD_CONTEXTKIT_DATABASE_URL not set — exiting")
        return 0

    asyncio.run(_set_field(client.project_id, "enable_graph", True))
    asyncio.run(_set_field(client.project_id, "custom_instructions", None))

    uid = f"qa-15-{uuid.uuid4().hex[:6]}"
    show("user_id", uid)

    try:
        # ── seed ──
        banner("0. Seed 4 memories")
        ids = []
        for fact in [
            "I love pizza and Italian food",
            "I work as a senior database engineer at Acme",
            "I drink black coffee every morning",
            "I'm allergic to peanuts",
        ]:
            out = client.add(
                messages=[{"role": "user", "content": fact}], user_id=uid
            )
            for r in (out or {}).get("results") or []:
                ids.append(r["id"])
        show("memory_ids", ids)
        if len(ids) < 3:
            print("\n❌ not enough rows — exiting")
            return 1
        A, B, C = ids[0], ids[1], ids[2]

        # ── versioning ──
        banner("1. Versioning — versions(A) / update×2 / rollback / latest_version")
        show("versions(A) before", client.versions(A))
        client.update(A, text="I love pizza, sushi, and Italian food")
        client.update(A, text="I love pizza, sushi, ramen, and Italian food")
        show("versions(A) after 2 updates", client.versions(A))
        show("latest_version(A)", client.latest_version(A))
        try:
            show("rollback(A, to_version=1)", client.rollback(A, to_version=1))
        except APIError as exc:
            show("rollback ERROR", f"status={exc.status} {exc}")
        show("versions(A) after rollback", client.versions(A))

        # ── MSPR feedback ──
        banner("2. MSPR feedback — POSITIVE on B, NEGATIVE on C")
        show(
            "feedback(B, POSITIVE)",
            client.feedback(B, feedback="POSITIVE", user_id=uid, query="job?"),
        )
        show(
            "feedback(C, NEGATIVE)",
            client.feedback(C, feedback="NEGATIVE", user_id=uid, query="morning?"),
        )

        # ── references ──
        banner("3. References — bump access counts on B and A")
        show("record_reference(B, 1.0)", client.record_reference(B, strength=1.0))
        show("record_reference(B, 0.5)", client.record_reference(B, strength=0.5))
        show("record_reference(A, 1.0)", client.record_reference(A, strength=1.0))

        time.sleep(0.5)

        # ── decay ──
        banner("4. Decay — tick / score / archive_low(threshold=0.5)")
        show("tick_decay", client.tick_decay())
        for label, mid in (("A", A), ("B", B), ("C", C)):
            show(f"decay_score({label})", client.decay_score(mid))
        show(
            "archive_low(threshold=0.5, dry_run=True)",
            client.archive_low(threshold=0.5, dry_run=True),
        )
        show(
            "archive_low(threshold=0.5, dry_run=False)",
            client.archive_low(threshold=0.5, dry_run=False),
        )

        # ── cold storage ──
        banner("5. Cold storage — demote + promote round-trip on C")
        try:
            show("demote_to_cold(C)", client.demote_to_cold(C))
            show("promote_from_cold(C)", client.promote_from_cold(C))
        except APIError as exc:
            show("cold ERROR", f"status={exc.status} {exc}")

        # ── backfill ──
        banner("6. Backfill — context_graph / mspr / decay_v2")
        for kind in ("context_graph", "mspr", "decay_v2"):
            try:
                show(f"backfill({kind!r})", client.backfill(kind))
            except APIError as exc:
                show(f"backfill({kind!r}) ERROR", f"status={exc.status} {exc}")

        # ── reset ──
        banner("7. reset_feedback(user_id) + reset_cgl")
        show("reset_feedback(user_id)", client.reset_feedback(user_id=uid))
        show("reset_cgl", client.reset_cgl())

    finally:
        try:
            client.delete_users(user_id=uid)
        except Exception:
            pass
        asyncio.run(_set_field(client.project_id, "enable_graph", False))
        client.close()

    banner("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
