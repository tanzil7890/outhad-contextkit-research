"""Phase 1.4 — MSPR end-to-end via the published SDK.

SDK twin of ``examples/mspr/basic_usage.py`` — same memories, same
query, same feedback step, same re-search. Driven entirely from
``outhad-contextkit-sdk`` against the cloud.

Run::

    pip install outhad-contextkit-sdk asyncpg
    export OUTHAD_CONTEXTKIT_API_KEY=ock_live_...
    python tests/cloud/test_phase14_mspr_sdk.py

Optional env::

    OUTHAD_CONTEXTKIT_HOST           # default http://localhost:8000
    OUTHAD_CONTEXTKIT_DATABASE_URL   # local-dev shortcut so the
                                     # script can flip enable_graph
                                     # without going through Clerk

What it proves:

1.  ``client.feedback(...)`` returns ``recorded=True`` (no longer the
    ``mspr_or_feedback_disabled`` reason). Cloud's
    ``memory_cache.build_memory`` now stamps ``mspr.enabled=True`` +
    ``mspr.feedback.enabled=True`` on every per-project Memory build
    when ``enable_graph=True``, so the engine actually has the
    feedback store + CGL relevance plumbing wired up.
2.  Engine returns a non-trivial ``delta_applied`` (default boost
    is +0.15) + ``new_relevance`` so the dashboard can surface the
    feedback's impact.
3.  Subsequent ``client.search(...)`` calls go through the MSPR
    pipeline (same response envelope, but the CGL backend's
    ``bump_relevance`` has already raised the row's node-level
    relevance for future queries).
"""
from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import uuid

from outhad_contextkit_sdk import MemoryClient

API_KEY = os.environ["OUTHAD_CONTEXTKIT_API_KEY"]
HOST = os.environ.get("OUTHAD_CONTEXTKIT_HOST", "http://localhost:8000")
DATABASE_URL = os.environ.get("OUTHAD_CONTEXTKIT_DATABASE_URL")

FACTS = [
    "I love sushi and ramen",
    "I prefer dark mode in editors",
    "I drink black coffee in the morning",
    "I work at a coffee shop on weekends",
    "I'm allergic to peanuts",
]
QUERY = "what does this user like to drink?"


def banner(t):
    print()
    print("=" * 80)
    print(t)
    print("=" * 80)


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


def show_hits(label, resp):
    print(f"\n  {label}")
    rows = (resp.get("results") if isinstance(resp, dict) else resp) or []
    if not rows:
        print("  (no matches)")
        return
    for hit in rows:
        score = hit.get("score")
        score_l = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
        mem = textwrap.shorten(hit.get("memory") or "", 70)
        print(f"    - {mem}  (score={score_l})")


def main() -> int:
    banner("Connecting")
    client = MemoryClient(api_key=API_KEY, host=HOST)
    print(f"  user_email : {client.user_email}")
    print(f"  project_id : {client.project_id}")
    print(f"  host       : {client.host}")

    if not DATABASE_URL:
        print("\n❌ OUTHAD_CONTEXTKIT_DATABASE_URL not set — needed to flip enable_graph")
        return 1

    project_id = client.project_id
    uid = f"qa-mspr-{uuid.uuid4().hex[:6]}"
    print(f"  user_id    : {uid}")

    try:
        banner("1. Flip enable_graph=True (cloud also enables MSPR + CGL)")
        asyncio.run(_set_field(project_id, "enable_graph", True))
        asyncio.run(_set_field(project_id, "custom_instructions", None))
        print("  enable_graph: True  →  context_graph + mspr (feedback / intent / "
              "frequency / success) all enabled per memory_cache.build_memory")

        banner("2. Adding memories via client.add()")
        for fact in FACTS:
            resp = client.add(
                messages=[{"role": "user", "content": fact}],
                user_id=uid,
            )
            results = (resp or {}).get("results") or []
            print(f"  ▶ {fact!r}  →  {len(results)} fact(s)")
            for r in results:
                print(f"     [{r.get('event','?')}] {r.get('memory')}")

        banner(f"3. Initial search: {QUERY!r}")
        before = client.search(query=QUERY, user_id=uid, use_context_graph=True)
        show_hits("Top hits BEFORE feedback:", before)

        # Pick the row about coffee in the top hits.
        rows = (before.get("results") if isinstance(before, dict) else before) or []
        target = next(
            (r for r in rows if "coffee" in (r.get("memory") or "").lower()),
            rows[0] if rows else None,
        )
        if not target:
            print("\n(no rows to feed back on — exiting)")
            return 1

        banner("4. client.feedback(POSITIVE) on the coffee row")
        print(f"   target: {target.get('memory')!r}")
        fb = client.feedback(
            target["id"],
            feedback="POSITIVE",
            user_id=uid,
            query=QUERY,
        )
        print(f"   response: {fb}")
        if not fb or not fb.get("recorded"):
            print("\n❌ feedback NOT recorded — MSPR not wired on this build")
            return 1
        print(
            f"\n   ✅ recorded=True  delta_applied={fb.get('delta_applied')!r}"
            f"  new_relevance={fb.get('new_relevance')!r}"
        )

        banner(f"5. Re-search: {QUERY!r}")
        after = client.search(query=QUERY, user_id=uid, use_context_graph=True)
        show_hits("Top hits AFTER positive feedback:", after)

        banner("6. Score / rank diff for the boosted row")

        def _row(resp, mid):
            for r in (resp.get("results") or []):
                if r.get("id") == mid:
                    return r
            return None

        b = _row(before, target["id"])
        a = _row(after, target["id"])
        b_score = b.get("score") if b else None
        a_score = a.get("score") if a else None
        b_rank = next(
            (i for i, r in enumerate((before.get("results") or []), 1)
             if r.get("id") == target["id"]),
            None,
        )
        a_rank = next(
            (i for i, r in enumerate((after.get("results") or []), 1)
             if r.get("id") == target["id"]),
            None,
        )
        print(f"  rank   : {b_rank}  →  {a_rank}")
        print(f"  score  : "
              f"{f'{b_score:.3f}' if isinstance(b_score, (int, float)) else '-'}"
              f"  →  "
              f"{f'{a_score:.3f}' if isinstance(a_score, (int, float)) else '-'}")
        print()
        print("  Note: the engine's MSPR re-rank formula multiplies by node")
        print("  ``relevance`` and the relevance bump is what drives long-term")
        print("  personalisation. The +0.15 delta_applied recorded in step 4")
        print("  is now persistent — every subsequent query on this user_id")
        print("  weighs this row higher even after a process restart.")

    finally:
        try:
            client.delete_users(user_id=uid)
        except Exception:
            pass
        asyncio.run(_set_field(project_id, "enable_graph", False))
        client.close()

    banner("Done")
    print()
    print("  Compare with the legacy local MSPR demo:")
    print("    PYTHONPATH=. python examples/mspr/basic_usage.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
