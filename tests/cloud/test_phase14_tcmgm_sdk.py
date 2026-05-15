"""Phase 1.4 — TCMGM end-to-end via the published SDK.

Verbose user-perspective walkthrough: every cloud call and every
response is printed in full so a developer can eyeball the wiring.

Run::

    pip install outhad-contextkit-sdk asyncpg
    export OUTHAD_CONTEXTKIT_API_KEY=ock_live_...
    python tests/cloud/test_phase14_tcmgm_sdk.py

Optional env::

    OUTHAD_CONTEXTKIT_HOST           # default http://localhost:8000
    OUTHAD_CONTEXTKIT_DATABASE_URL   # local-dev shortcut so the script
                                     # can flip enable_graph without
                                     # going through Clerk
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import textwrap
import uuid

from outhad_contextkit_sdk import MemoryClient

API_KEY = os.environ["OUTHAD_CONTEXTKIT_API_KEY"]
HOST = os.environ.get("OUTHAD_CONTEXTKIT_HOST", "http://localhost:8000")
DATABASE_URL = os.environ.get("OUTHAD_CONTEXTKIT_DATABASE_URL")


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def show(label: str, value) -> None:
    """Pretty-print a labelled value — dicts/lists go through json."""
    print(f"\n  ▸ {label}:")
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, indent=2, default=str)
        print(textwrap.indent(rendered, "    "))
    else:
        print(f"    {value}")


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


def main() -> int:
    banner("Connecting")
    client = MemoryClient(api_key=API_KEY, host=HOST)
    show("user_email", client.user_email)
    show("project_id", client.project_id)
    show("host", client.host)

    # ── 1. SDK signature surface ──────────────────────────────────────
    banner("1. SDK 1.4 search() signature exposes TCMGM kwargs")
    sig = inspect.signature(client.search)
    needed = {"use_tcmgm", "time_window", "include_causal"}
    present = needed & set(sig.parameters.keys())
    show("expected kwargs", sorted(needed))
    show("present in SDK", sorted(present))
    show("missing", sorted(needed - present) or "—")

    if not DATABASE_URL:
        banner("Skipping graph-toggle tests")
        print("\n  OUTHAD_CONTEXTKIT_DATABASE_URL not set — without it the")
        print("  script can't flip ``enable_graph`` for the TCMGM-on path.")
        print("  Set it to your Neon connection string to run the full suite.")
        client.close()
        return 0

    # ── 2. graceful no-op when graph off ──────────────────────────────
    banner("2. use_tcmgm=True with enable_graph=False (graceful no-op)")
    asyncio.run(_set_field(client.project_id, "enable_graph", False))
    asyncio.run(_set_field(client.project_id, "custom_instructions", None))
    uid = f"qa-tcmgm-{uuid.uuid4().hex[:6]}"
    print(f"\n  user_id: {uid}")
    print("  → client.add(messages=[{...}], user_id=uid)")
    add_resp = client.add(
        messages=[{"role": "user", "content": f"Coffee preference for {uid}."}],
        user_id=uid,
    )
    show("add() response", add_resp)
    print("\n  → client.search(query=..., user_id=uid, use_tcmgm=True)")
    search_resp = client.search(
        query="What does this user prefer?",
        user_id=uid,
        use_tcmgm=True,
    )
    show("search response keys", sorted(search_resp.keys()) if isinstance(search_resp, dict) else type(search_resp).__name__)
    show("search response", search_resp)
    try:
        client.delete_users(user_id=uid)
    except Exception:
        pass

    # ── 3. custom_instructions doesn't break add ──────────────────────
    banner("3. custom_instructions no longer crashes add() (FACT_RETRIEVAL_PROMPT append fix)")
    asyncio.run(_set_field(
        client.project_id,
        "custom_instructions",
        "Always be concise; cite the source whenever possible.",
    ))
    show("custom_instructions set on project", "Always be concise; cite the source whenever possible.")
    uid = f"qa-ci-{uuid.uuid4().hex[:6]}"
    print(f"\n  user_id: {uid}")
    print("  → client.add(...)")
    add_resp = client.add(
        messages=[{"role": "user", "content": f"Custom-instr smoke {uid}."}],
        user_id=uid,
    )
    show("add() response", add_resp)
    asyncio.run(_set_field(client.project_id, "custom_instructions", None))
    try:
        client.delete_users(user_id=uid)
    except Exception:
        pass

    # ── 4. TCMGM fused envelope ───────────────────────────────────────
    banner("4. use_tcmgm=True with enable_graph=True returns the fused envelope")
    asyncio.run(_set_field(client.project_id, "enable_graph", True))
    show("enable_graph", True)
    uid = f"qa-tcmgm-g-{uuid.uuid4().hex[:6]}"
    print(f"\n  user_id: {uid}")
    print("  → client.add(...) (1 of 2)")
    client.add(
        messages=[{"role": "user", "content": f"I drink coffee every morning ({uid})."}],
        user_id=uid,
    )
    print("  → client.add(...) (2 of 2)")
    client.add(
        messages=[{"role": "user", "content": f"I prefer dark mode editors ({uid})."}],
        user_id=uid,
    )
    print("  → client.search(query='morning routine?', use_tcmgm=True, include_causal=True)")
    hits = client.search(
        query="morning routine?",
        user_id=uid,
        use_tcmgm=True,
        include_causal=True,
    )
    show("response keys", sorted(hits.keys()) if isinstance(hits, dict) else type(hits).__name__)
    show("fused_ranking", hits.get("fused_ranking") if isinstance(hits, dict) else None)
    show("timeline_results", hits.get("timeline_results") if isinstance(hits, dict) else None)
    show("causal_chains", hits.get("causal_chains") if isinstance(hits, dict) else None)
    show("graph_results", hits.get("graph_results") if isinstance(hits, dict) else None)

    # ── 5. time_window + include_causal passthrough ───────────────────
    banner("5. use_tcmgm + time_window + include_causal=False passthrough")
    print("  → client.search(use_tcmgm=True, time_window={...}, include_causal=False)")
    hits2 = client.search(
        query="dark mode?",
        user_id=uid,
        use_tcmgm=True,
        time_window={"start": "2020-01-01T00:00:00Z", "end": "2030-12-31T23:59:59Z"},
        include_causal=False,
    )
    show("response keys", sorted(hits2.keys()) if isinstance(hits2, dict) else type(hits2).__name__)
    show("fused_ranking", hits2.get("fused_ranking") if isinstance(hits2, dict) else None)
    show("causal_chains (should be empty since include_causal=False)",
         hits2.get("causal_chains") if isinstance(hits2, dict) else None)

    try:
        client.delete_users(user_id=uid)
    except Exception:
        pass
    asyncio.run(_set_field(client.project_id, "enable_graph", False))

    banner("Done")
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
