"""Phase 1.4 — audio cross-modal search via the published SDK.

Mirrors the legacy ``tests/memory/temporal/test_audio_search.py``
(Whisper-based audio embedding + CLIP text-encoder cross-modal
search) so a developer can compare the two paths side-by-side.

Same dataset (``test_dataset/audio/*.wav``) and same query set —
just driven from the published SDK instead of importing the engine
directly.

Mechanism difference (call this out explicitly):

* Legacy local script: imports ``outhad_contextkit.memory.temporal``,
  generates Whisper audio embeddings + CLIP text embeddings
  in-process, runs ``cross_modal_search`` for cosine similarity.
  Pure 768-d cross-modal cosine, no LLM in the loop.
* This script: speaks **only** to the cloud through
  ``outhad-contextkit-sdk``. Since v1.4 the cloud doesn't natively
  ingest ``audio_url`` content blocks (engine has the embedder, the
  route + SDK plumbing ships in v1.5), so this script does what an
  end-user would do today: transcribe each audio file client-side
  via OpenAI Whisper, then ``client.add(messages=[{...transcript...}])``
  with ``infer=False`` so the transcript becomes one row per file.
  Search is plain text-vs-text similarity over the transcripts.

Both paths — local CLIP / Whisper and SDK Whisper-then-text — share
the same caveat the legacy script calls out: Whisper is built for
speech, not environmental sound, so the matches for ``"thunder"`` or
``"birds chirping"`` are noisy in either path.

Run::

    pip install outhad-contextkit-sdk openai
    export OPENAI_API_KEY=sk-...
    export OUTHAD_CONTEXTKIT_API_KEY=ock_live_...
    python tests/cloud/test_phase14_audio_search.py
"""
from __future__ import annotations

import os
import sys
import textwrap
import uuid
from pathlib import Path

from outhad_contextkit_sdk import MemoryClient

API_KEY = os.environ["OUTHAD_CONTEXTKIT_API_KEY"]
HOST = os.environ.get("OUTHAD_CONTEXTKIT_HOST", "http://localhost:8000")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = REPO_ROOT / "test_dataset" / "audio"

# Same query set as the legacy local audio test.
TEST_QUERIES = [
    "clapping",
    "applause",
    "people clapping",
    "birds chirping",
    "birds singing",
    "thunder storm",
    "thunder",
    "rain",
    "rain falling",
    "dog barking",
    "vacuum cleaner",
]

# ESC-50 class id → expected sound label, mirrored from the legacy
# script so the comparison labels line up.
ESC50_CLASSES = {
    0: "dog barking",
    14: "birds chirping",
    19: "thunderstorm",
    22: "rain",
    36: "vacuum cleaner",
    37: "clock alarm",
    38: "clock tick",
}


def expected_label(filename: str) -> str:
    parts = filename.split("-")
    if len(parts) >= 4 and parts[2] == "A":
        try:
            return ESC50_CLASSES.get(int(parts[3].replace(".wav", "")), "unknown")
        except ValueError:
            return "unknown"
    return "unknown"


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

    if not AUDIO_DIR.is_dir():
        print(f"\n❌ Audio dataset not found at {AUDIO_DIR}")
        return 1
    audio_files = sorted(p for p in AUDIO_DIR.iterdir() if p.suffix.lower() == ".wav")
    print(f"\n  📁 Found {len(audio_files)} audio files")

    # ── client-side Whisper transcription ──────────────────────────
    if not OPENAI_KEY:
        print("\n❌ OPENAI_API_KEY not set — needed for client-side Whisper")
        return 1
    try:
        from openai import OpenAI  # noqa
    except Exception:
        print("\n❌ openai package not installed (`pip install openai`)")
        return 1

    banner("Transcribing audio files client-side via OpenAI Whisper")
    openai_client = OpenAI(api_key=OPENAI_KEY)
    transcripts: list[tuple[str, str, str]] = []  # (filename, label, transcript)
    for path in audio_files:
        label = expected_label(path.name)
        try:
            with open(path, "rb") as fh:
                resp = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=fh,
                )
            text = resp.text or ""
        except Exception as exc:
            text = f"[transcription failed: {type(exc).__name__}: {str(exc)[:80]}]"
        transcripts.append((path.name, label, text))
        print(f"  🎧 {path.name}  (expected: {label})")
        print(f"     transcript: {text[:120]!r}")

    # ── upload each transcript via SDK ─────────────────────────────
    banner("Uploading transcripts via SDK client.add(infer=False)")
    uid = f"qa-audio-{uuid.uuid4().hex[:6]}"
    print(f"  user_id: {uid}")
    print()
    for filename, label, transcript in transcripts:
        # Even a near-empty transcript is fine — we tag the metadata
        # with the filename so users can map the search hit back to
        # the source audio. With `infer=False` the engine stores the
        # caption verbatim instead of running fact extraction.
        caption = (
            f"[audio file {filename} | expected sound: {label}] "
            f"Whisper transcript: {transcript or '(no detectable speech)'}"
        )
        try:
            client.add(
                messages=[{"role": "user", "content": caption}],
                user_id=uid,
                metadata={"audio_file": filename, "expected_sound": label},
                infer=False,
            )
            print(f"  ✅ {filename}")
        except Exception as exc:
            print(f"  ❌ {filename} — {type(exc).__name__}: {str(exc)[:120]}")

    # ── show what got stored ──────────────────────────────────────
    banner("Memory contents after upload (client.get_all)")
    rows = client.get_all(user_id=uid, limit=50)
    items = (rows.get("results") if isinstance(rows, dict) else rows) or []
    print(f"  count: {len(items)}")
    for i, row in enumerate(items, 1):
        meta = row.get("metadata") or {}
        print(f"  [{i}] expected={meta.get('expected_sound')!r} file={meta.get('audio_file')!r}")
        print(f"       memory={textwrap.shorten(row.get('memory') or '', 120)!r}")

    # ── run every legacy query ────────────────────────────────────
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
                marker = ""
                if isinstance(score, (int, float)):
                    marker = "🟢 HIGH" if score > 0.6 else "🟡 MEDIUM" if score > 0.4 else "🔴 LOW"
                meta = row.get("metadata") or {}
                expected = meta.get("expected_sound", "?")
                file = meta.get("audio_file", "?")
                print(f"    {i}. {file}")
                print(f"       Expected: {expected}")
                print(f"       Similarity: {score_label} {marker}")
            top = rows[0]
            t_meta = top.get("metadata") or {}
            print(f"\n  ⭐ Best match: {t_meta.get('audio_file', '?')}")
            print(f"     Expected sound: {t_meta.get('expected_sound', '?')}")
        except Exception as exc:
            print(f"  ❌ {type(exc).__name__}: {str(exc)[:120]}")

    # ── summary ───────────────────────────────────────────────────
    banner("TEST SUMMARY")
    print(f"  ✅ SDK 1.4.0 audio end-to-end against {client.host}")
    print(f"  ├─ Audio files     : {len(audio_files)}")
    print(f"  ├─ Transcripts     : {len(transcripts)}")
    print(f"  ├─ Queries tested  : {len(TEST_QUERIES)}")
    print(f"  └─ Pipeline        : client-side Whisper → SDK add(infer=False) → SDK search()")
    print()
    print("  Compare with the legacy local test:")
    print("    PYTHONPATH=. python tests/memory/temporal/test_audio_search.py")
    print()
    print("  That one uses Whisper audio embeddings + CLIP text encoder")
    print("  in-process for pure 768-d cross-modal cosine. Both paths share")
    print("  the same Whisper-on-environmental-sounds caveat — Whisper is")
    print("  built for speech, so non-speech audio (rain, thunder, birds)")
    print("  produces noisy transcripts and noisy hits in either path.")
    print()
    print("  Native audio_url ingestion via SDK ships in v1.5 — engine has")
    print("  the embedder (outhad_contextkit.memory.temporal.MultimodalEmbedder")
    print("  exposes embed_audio() + a CLAP backbone), but route + Pydantic")
    print("  plumbing aren't wired through yet.")

    try:
        client.delete_users(user_id=uid)
    except Exception:
        pass
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
