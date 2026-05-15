"""Phase 1.6 — pin / hybrid / LLM filter / advanced filters / idempotency.

Verbose user-perspective walkthrough against published SDK 1.6.0.

Run::

    pip install outhad-contextkit-sdk
    export OUTHAD_CONTEXTKIT_API_KEY=ock_live_...
    python tests/cloud/test_phase16_extras.py
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
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
    banner("Connecting")
    client = MemoryClient(api_key=API_KEY, host=HOST)
    show("user_email", client.user_email)
    show("project_id", client.project_id)

    uid = f"qa-16-{uuid.uuid4().hex[:6]}"
    show("user_id", uid)

    try:
        # ── seed memories ───────────────────────────────────
        banner("0. Seed 5 memories with varied metadata")
        ids = []
        seeds = [
            ("I love pizza and Italian food", {"category": "food", "priority": 5}),
            ("I work at Acme as senior database engineer", {"category": "job", "priority": 8}),
            ("I drink black coffee every morning", {"category": "food", "priority": 3}),
            ("I'm allergic to peanuts", {"category": "health", "priority": 9}),
            ("I prefer dark mode editors", {"category": "tools", "priority": 2}),
        ]
        for fact, meta in seeds:
            out = client.add(
                messages=[{"role": "user", "content": fact}],
                user_id=uid, metadata=meta,
            )
            for r in (out or {}).get("results") or []:
                ids.append((r["id"], r["memory"]))
        show("memory_ids", ids)
        if not ids:
            return 1
        target_id = ids[0][0]

        # ── 1. Pin / lock ────────────────────────────────────
        banner("1. Pin / lock — update + delete should 409 on locked rows")
        show("pin(target)", client.pin(target_id))
        try:
            client.update(target_id, text="trying to overwrite locked row")
            show("update on locked → ", "❌ NO 409 — bug")
        except APIError as exc:
            show("update on locked → expected 409", f"status={exc.status} {str(exc)[:80]}")
        try:
            client.delete(target_id)
            show("delete on locked → ", "❌ NO 409 — bug")
        except APIError as exc:
            show("delete on locked → expected 409", f"status={exc.status} {str(exc)[:80]}")
        show("unpin(target)", client.unpin(target_id))
        # Should now succeed
        client.update(target_id, text="updated after unpin")
        show("update after unpin", "✅ ok")

        # ── 2. Advanced filter operators ────────────────────
        banner("2. Advanced filter operators ($gt / $in / AND / OR)")
        # priority > 5 → job (8), health (9)
        out_gt = client.get_all_filtered(
            {"metadata.priority": {"$gt": 5}},
            user_id=uid, limit=20,
        )
        show("priority > 5", out_gt)
        # category in [food, tools] → food rows + tools row
        out_in = client.get_all_filtered(
            {"metadata.category": {"$in": ["food", "tools"]}},
            user_id=uid, limit=20,
        )
        show("category in [food, tools]", out_in)
        # AND: priority>=5 AND category=food
        out_and = client.get_all_filtered(
            {"AND": [
                {"metadata.priority": {"$gte": 5}},
                {"metadata.category": "food"},
            ]},
            user_id=uid, limit=20,
        )
        show("AND priority>=5 AND category=food", out_and)
        # OR
        out_or = client.get_all_filtered(
            {"OR": [
                {"metadata.priority": {"$lt": 3}},
                {"metadata.category": "health"},
            ]},
            user_id=uid, limit=20,
        )
        show("OR priority<3 OR category=health", out_or)

        # ── 3. Hybrid keyword search ────────────────────────
        banner("3. Hybrid keyword search (keyword_search=True)")
        rare = client.search(query="Acme", user_id=uid, limit=5)
        show("plain vector search 'Acme'", rare)
        rare_h = client.search(query="Acme", user_id=uid, limit=5, keyword_search=True)
        show("hybrid keyword_search=True 'Acme'", rare_h)

        # ── 4. LLM filter step ──────────────────────────────
        banner("4. LLM relevance gate (filter_memories=True)")
        before = client.search(
            query="What food do I like?", user_id=uid, limit=5,
        )
        show("plain search 'What food do I like?'", before)
        gated = client.search(
            query="What food do I like?", user_id=uid, limit=5, filter_memories=True,
        )
        show("filter_memories=True", gated)

        # ── 5. Idempotency ──────────────────────────────────
        banner("5. Idempotency-Key — same key returns cached response")
        idem = f"key-{uuid.uuid4().hex[:8]}"
        first = client.add_idempotent(
            idem,
            messages=[{"role": "user", "content": "Idempotency smoke test row"}],
            user_id=uid,
        )
        show("first call (idem={!r})".format(idem), first)
        second = client.add_idempotent(
            idem,
            messages=[{"role": "user", "content": "Idempotency smoke test row"}],
            user_id=uid,
        )
        show("second call (same key) — should be byte-identical", second)
        match = first == second
        show("first == second", match)

    finally:
        try:
            # Ensure target is unpinned before delete_users
            client.unpin(target_id)
        except Exception:
            pass
        try:
            client.delete_users(user_id=uid)
        except Exception:
            pass
        client.close()

    banner("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
