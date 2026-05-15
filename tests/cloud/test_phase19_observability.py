"""Phase 1.9 — customer observability via published SDK 1.9.0.

Drives every Phase 1.9 surface against PyPI install:
  - Usage summary (counts / latency / status mix)
  - Request log trace viewer (filters: path / status / user_id)
  - Top users panel (group by user_id_scope)
  - Graph snapshot (CGL nodes + edges or enabled=False explanation)

Run::

    /tmp/sdk19_test/bin/pip install outhad-contextkit-sdk==1.9.0
    export OUTHAD_CONTEXTKIT_API_KEY=ock_live_...
    export OUTHAD_CONTEXTKIT_HOST=http://localhost:8000
    /tmp/sdk19_test/bin/python tests/cloud/test_phase19_observability.py
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import time
import uuid

from outhad_contextkit_sdk import APIError, MemoryClient

API_KEY = os.environ["OUTHAD_CONTEXTKIT_API_KEY"]
HOST = os.environ.get("OUTHAD_CONTEXTKIT_HOST", "http://localhost:8000")


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


def main() -> int:
    banner("Connecting via SDK 1.9.0")
    client = MemoryClient(api_key=API_KEY, host=HOST)
    show("user_email", client.user_email)
    show("project_id", client.project_id)

    # Generate fresh signal so the dashboards aren't empty.
    suffix = uuid.uuid4().hex[:6]
    qa_user_a = f"obs-qa-A-{suffix}"
    qa_user_b = f"obs-qa-B-{suffix}"

    banner("0. Seed traffic — drive a few add/search/get calls so the trace viewer has rows")
    try:
        a1 = client.add(
            messages=[{"role": "user", "content": "Mocha latte is my morning order."}],
            user_id=qa_user_a,
        )
        show("add A", a1)
        a2 = client.add(
            messages=[{"role": "user", "content": "I prefer window seats on long-haul flights."}],
            user_id=qa_user_b,
        )
        show("add B", a2)
        # Two searches under the same user for the top-users panel.
        client.search(query="favorite drink", user_id=qa_user_a)
        client.search(query="seat preference", user_id=qa_user_b)
        client.search(query="seat preference", user_id=qa_user_b)
        # And a deliberate 404 to exercise the failed-request filter.
        try:
            client.get("11111111-1111-1111-1111-111111111111")
        except APIError as exc:
            show("expected 404 on bogus get", f"status={exc.status}")
    except APIError as exc:
        show("seed traffic ERROR", f"status={exc.status} {exc}")

    # Give the middleware a moment to commit.
    time.sleep(1.5)

    # ── 1. Usage summary ──────────────────────────────────────────────
    banner("1. client.usage() — aggregate counters / latency / status mix")
    summary = client.usage()
    show("usage summary", summary)

    # ── 2. Trace viewer ───────────────────────────────────────────────
    banner("2. client.requests(limit=10) — newest first")
    page = client.requests(limit=10)
    show(f"requests (showing {len(page.get('results') or [])} of total={page.get('total')})", page)

    banner("2b. client.requests(min_status=400, limit=5) — failed-request filter")
    fails = client.requests(min_status=400, limit=5)
    show("failed-request page", fails)

    banner(f"2c. client.requests(user_id={qa_user_b!r}, limit=10) — drill-down by user")
    by_user = client.requests(user_id=qa_user_b, limit=10)
    show(f"requests for {qa_user_b}", by_user)

    # ── 3. Top users ──────────────────────────────────────────────────
    banner("3. client.top_users(limit=5)")
    top = client.top_users(limit=5)
    show("top users", top)

    # ── 4. Graph snapshot ─────────────────────────────────────────────
    banner("4. client.graph_snapshot(limit=50)")
    graph = client.graph_snapshot(limit=50)
    keys = sorted((graph or {}).keys())
    show("graph keys", keys)
    show(
        "graph stats",
        {
            "enabled": graph.get("enabled"),
            "nodes": len(graph.get("nodes") or []),
            "edges": len(graph.get("edges") or []),
            "truncated": graph.get("truncated"),
            "detail": graph.get("detail"),
        },
    )
    if graph.get("nodes"):
        show("first node", (graph.get("nodes") or [None])[0])
    if graph.get("edges"):
        show("first edge", (graph.get("edges") or [None])[0])

    # ── 5. Path-filter sanity ─────────────────────────────────────────
    banner("5. client.requests(path='/v1/usage', limit=3) — meta filter")
    meta = client.requests(path="/v1/usage", limit=3)
    show("requests touching /v1/usage", meta)

    client.close()
    banner("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
