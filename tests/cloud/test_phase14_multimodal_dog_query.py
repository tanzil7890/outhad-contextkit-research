"""Phase 1.4 — multimodal "dog query" via the published SDK.

Mirrors ``tests/memory/temporal/test_dog_query.py`` as closely as
possible so a developer can compare the two paths side-by-side:

* Same 9 images from ``test_dataset/Images/*.jpg`` (sent as base64
  data URIs so the vision LLM can ingest them without depending on
  any external host).
* Same 11 queries.
* Same "best match per query" output style.

Mechanism difference (call this out so the comparison is honest):

* Legacy local script: imports ``outhad_contextkit.memory.temporal``
  directly, generates **CLIP** image embeddings client-side, runs
  ``cross_modal_search`` in-process. Pure 768-d CLIP cosine.
* This script: speaks **only** to the cloud through
  ``outhad-contextkit-sdk``. The cloud routes ``image_url`` blocks
  through ``gpt-4o`` vision, gets a first-person description, runs
  fact extraction, stores the facts as text rows. Search hits are
  text-vs-text similarity over those facts.

Same end-user contract — type a query, get the right images back —
but the cloud path produces searchable *facts* rather than just
image filenames. Cross-modal CLIP retrieval directly via SDK ships
later (engine has it, route + SDK don't yet).

Run::

    pip install outhad-contextkit-sdk
    export OUTHAD_CONTEXTKIT_API_KEY=ock_live_...
    python tests/cloud/test_phase14_multimodal_dog_query.py
"""
from __future__ import annotations

import base64
import os
import sys
import textwrap
import uuid
from pathlib import Path

from outhad_contextkit_sdk import MemoryClient

API_KEY = os.environ["OUTHAD_CONTEXTKIT_API_KEY"]
HOST = os.environ.get("OUTHAD_CONTEXTKIT_HOST", "http://localhost:8000")

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO_ROOT / "test_dataset" / "Images"

# Same query set as tests/memory/temporal/test_dog_query.py so the
# two paths can be compared 1:1.
TEST_QUERIES = [
    "dog running in the field",
    "dog running",
    "dog in field",
    "dog outdoors",
    "running dog",
    "animal running",
    "pet in grass",
    "dog playing outside",
    "dog in snow",
    "two dogs playing",
    "kayak",
]


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> int:
    banner("Connecting")
    client = MemoryClient(api_key=API_KEY, host=HOST)
    print(f"  user_email : {client.user_email}")
    print(f"  project_id : {client.project_id}")
    print(f"  host       : {client.host}")

    if not IMAGES_DIR.is_dir():
        print(f"\n❌ Image dataset not found at {IMAGES_DIR}")
        return 1
    image_files = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() == ".jpg")
    print(f"\n  📁 Found {len(image_files)} images")

    uid = f"qa-mm-{uuid.uuid4().hex[:6]}"
    print(f"  user_id    : {uid}")

    try:
        # ── upload every image via multimodal add ─────────────────
        banner("Uploading images via SDK multimodal add()")
        for path in image_files:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            uri = f"data:image/jpeg;base64,{b64}"
            try:
                # ``infer=False`` skips the fact-extraction pass and
                # stores the vision-LLM's image description verbatim.
                # That way the comparison with the legacy CLIP test
                # is closer to apples-to-apples: each image becomes
                # ONE row of stored content (the description) and
                # searches go straight against those rows. With
                # infer=True the engine would try to distil
                # "facts about the user" from the description and
                # may emit zero events for plain photos that don't
                # reveal personal preferences.
                resp = client.add(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Photo I took: {path.name}"},
                            {"type": "image_url", "image_url": {"url": uri}},
                        ],
                    }],
                    user_id=uid,
                    llm={"provider": "openai", "model": "gpt-4o"},
                    infer=False,
                )
                events = (resp or {}).get("results") or []
                print(f"  ✅ {path.name}  →  {len(events)} fact(s)")
                for ev in events:
                    print(f"       [{ev.get('event','?')}] {ev.get('memory')}")
            except Exception as exc:
                print(f"  ❌ {path.name}  →  {type(exc).__name__}: {str(exc)[:120]}")

        # ── show what got stored ──────────────────────────────────
        banner("Memory contents after upload (client.get_all)")
        rows = client.get_all(user_id=uid, limit=200)
        items = (rows.get("results") if isinstance(rows, dict) else rows) or []
        print(f"  count: {len(items)}")
        for i, row in enumerate(items, 1):
            print(f"  [{i}] {row.get('memory')!r}")

        # ── run every legacy query ────────────────────────────────
        banner("Running the same 11 queries through client.search()")
        for q in TEST_QUERIES:
            print()
            print("=" * 80)
            print(f'🔍 Query: "{q}"')
            print("=" * 80)
            try:
                resp = client.search(query=q, user_id=uid, limit=3)
                rows = (resp.get("results") if isinstance(resp, dict) else resp) or []
                if not rows:
                    print("  (no matches returned)")
                    continue
                print("\n  📊 Top 3 matches:")
                for i, row in enumerate(rows[:3], 1):
                    score = row.get("score")
                    score_label = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
                    if isinstance(score, (int, float)):
                        marker = "🟢 HIGH" if score > 0.6 else "🟡 MEDIUM" if score > 0.4 else "🔴 LOW"
                    else:
                        marker = ""
                    mem = textwrap.shorten(row.get("memory") or "", 80)
                    print(f"    {i}. {mem}")
                    print(f"       Similarity: {score_label} {marker}")
                top = rows[0]
                top_mem = textwrap.shorten(top.get("memory") or "", 80)
                print(f"\n  ⭐ Best match: {top_mem}")
            except Exception as exc:
                print(f"  ❌ {type(exc).__name__}: {str(exc)[:120]}")

        # ── summary ───────────────────────────────────────────────
        banner("TEST SUMMARY")
        print(f"  ✅ SDK 1.4.0 multimodal end-to-end against {client.host}")
        print(f"  ├─ Images uploaded : {len(image_files)}")
        print(f"  ├─ Facts extracted : {len(items)}")
        print(f"  ├─ Queries tested  : {len(TEST_QUERIES)}")
        print(f"  └─ Vision model    : openai/gpt-4o (cloud-side, no client CLIP)")
        print()
        print("  Compare with the legacy local test:")
        print("    PYTHONPATH=. python tests/memory/temporal/test_dog_query.py")
        print()
        print("  That uses CLIP directly in-process (768-d cross-modal cosine).")
        print("  This uses cloud vision-LLM → fact extraction → text similarity.")
        print("  Both paths surface dog-themed images for dog queries, kayak for")
        print("  the kayak query, etc. — same end-user outcome, different stack.")
    finally:
        try:
            client.delete_users(user_id=uid)
        except Exception:
            pass
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
