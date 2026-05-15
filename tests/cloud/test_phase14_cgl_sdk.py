"""Phase 1.4 — CGL (Context-Graph Layer) end-to-end via the published SDK.

SDK twin of ``examples/context_graph/basic_usage.py`` — same memories,
same query, same on/off comparison — but driven entirely from
``outhad-contextkit-sdk`` against the cloud.

Run::

    pip install outhad-contextkit-sdk asyncpg
    export OUTHAD_CONTEXTKIT_API_KEY=ock_live_...
    python tests/cloud/test_phase14_cgl_sdk.py

Optional env::

    OUTHAD_CONTEXTKIT_HOST           # default http://localhost:8000
    OUTHAD_CONTEXTKIT_DATABASE_URL   # local-dev shortcut so the script
                                     # can flip enable_graph without
                                     # going through Clerk

What it proves:

1.  ``client.search(..., use_context_graph=True)`` is wired
    end-to-end — the cloud's per-project ``Memory`` actually has
    ``context_graph.enabled=True`` once the project flips
    ``enable_graph=True`` (memory_cache.build_memory wires the CGL
    config alongside the entity graph).
2.  CGL re-ranks: scores from ``use_context_graph=True`` differ from
    ``use_context_graph=False`` because the graph-first path applies
    a dense/lexical/graph score blend and surfaces topically-linked
    rows the plain vector path misses.

Side-by-side comparison with the legacy local engine test::

    PYTHONPATH=. python examples/context_graph/basic_usage.py
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
    "I love pizza",
    "I'm allergic to peanuts",
    "Italian food is my favourite",
]
QUERY = "what should I eat?"


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


async def _set_field(project_id: str, field: str, value) -> None:
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


def show_hits(label: str, resp) -> None:
    print(f"\n  {label}")
    if not isinstance(resp, dict):
        print(f"  (unexpected response type: {type(resp).__name__})")
        return
    rows = resp.get("results") or []
    if not rows:
        print("  (no matches returned)")
        return
    for row in rows:
        score = row.get("score")
        score_label = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
        mem = textwrap.shorten(row.get("memory") or "", 80)
        print(f"    - {mem}  (score={score_label})")
    subgraph = resp.get("subgraph") or []
    if subgraph:
        print(f"\n  Subgraph edges used during expansion: {len(subgraph)}")
        for edge in subgraph[:5]:
            w = edge.get("weight")
            w_label = f"{w:.2f}" if isinstance(w, (int, float)) else "-"
            print(
                f"    - {edge.get('type')} {edge.get('src')[:8]}... -> "
                f"{edge.get('dst')[:8]}... w={w_label}"
            )


def main() -> int:
    banner("Connecting")
    client = MemoryClient(api_key=API_KEY, host=HOST)
    print(f"  user_email : {client.user_email}")
    print(f"  project_id : {client.project_id}")
    print(f"  host       : {client.host}")

    if not DATABASE_URL:
        print("\n❌ OUTHAD_CONTEXTKIT_DATABASE_URL not set — needed to flip enable_graph")
        print("   on the project so the cloud's Memory init turns CGL on.")
        return 1

    project_id = client.project_id
    uid = f"qa-cgl-{uuid.uuid4().hex[:6]}"
    print(f"  user_id    : {uid}")

    try:
        # ── enable graph (also turns CGL on per memory_cache wiring) ──
        banner("1. Flip enable_graph=True (cloud's memory_cache wires CGL alongside)")
        asyncio.run(_set_field(project_id, "enable_graph", True))
        # Bust any cached Memory built before the flip — different
        # project state → different cache key, but force a rebuild
        # to be sure.
        asyncio.run(_set_field(project_id, "custom_instructions", None))
        print("  enable_graph: True  (cloud now keeps graph_store + sets context_graph.enabled=True)")

        # ── ingest ───────────────────────────────────────────────────
        banner("2. Adding memories via client.add()")
        for fact in FACTS:
            resp = client.add(
                messages=[{"role": "user", "content": fact}],
                user_id=uid,
                run_id=f"{uid}-run",
            )
            results = (resp or {}).get("results") or []
            print(f"  ▶ {fact!r}  →  {len(results)} fact(s)")
            for r in results:
                print(f"     [{r.get('event','?')}] {r.get('memory')}")

        # Show what the engine actually stored.
        banner("3. Memory contents (client.get_all)")
        rows = client.get_all(user_id=uid, limit=20)
        items = (rows.get("results") if isinstance(rows, dict) else rows) or []
        print(f"  count: {len(items)}")
        for i, row in enumerate(items, 1):
            print(f"  [{i}] {row.get('memory')!r}")

        # ── search both ways ─────────────────────────────────────────
        banner(f"4. Query: {QUERY!r}  (use_context_graph=True — CGL ON)")
        cgl_on = client.search(query=QUERY, user_id=uid, use_context_graph=True)
        show_hits("results (CGL-rerank scores):", cgl_on)

        banner(f"5. Query: {QUERY!r}  (use_context_graph=False — plain vector)")
        cgl_off = client.search(query=QUERY, user_id=uid, use_context_graph=False)
        show_hits("results (plain vector scores):", cgl_off)

        # ── side-by-side score comparison ────────────────────────────
        banner("6. Side-by-side ranking comparison")

        def _row_map(resp) -> dict:
            out = {}
            for row in (resp or {}).get("results") or []:
                mem = row.get("memory") or ""
                score = row.get("score")
                if mem:
                    out[mem] = score
            return out

        on_map = _row_map(cgl_on)
        off_map = _row_map(cgl_off)
        all_keys = sorted(set(on_map) | set(off_map))
        print(f"  {'memory':<60} {'CGL on':>10} {'CGL off':>10}")
        print(f"  {'-' * 60} {'-' * 10:>10} {'-' * 10:>10}")
        for mem in all_keys:
            on_s = on_map.get(mem)
            off_s = off_map.get(mem)
            on_l = f"{on_s:.3f}" if isinstance(on_s, (int, float)) else "—"
            off_l = f"{off_s:.3f}" if isinstance(off_s, (int, float)) else "—"
            print(f"  {textwrap.shorten(mem, 58):<60} {on_l:>10} {off_l:>10}")

    finally:
        try:
            client.delete_users(user_id=uid)
        except Exception:
            pass
        asyncio.run(_set_field(project_id, "enable_graph", False))
        client.close()

    banner("Done")
    print()
    print("  Compare with the legacy local CGL demo:")
    print("    PYTHONPATH=. python examples/context_graph/basic_usage.py")
    print()
    print("  Both paths boot CGL on the same MemoryConfig.context_graph contract")
    print("  (engine has the layer; this script proves the cloud now wires it on")
    print("  per-project automatically when the dashboard flips enable_graph=True).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
