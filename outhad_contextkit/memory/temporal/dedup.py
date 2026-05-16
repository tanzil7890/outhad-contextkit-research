""" Event de-duplication.

Conversational data frequently re-states the same fact across turns
(echo questions, paraphrased confirmations, retries). Without dedup
the timeline + causal graph end up with multiple near-identical
nodes that fragment retrieval and inflate downstream LLM costs.

Strategy: pairwise similarity over canonical event content using
``difflib.SequenceMatcher`` (stdlib — zero new deps). Industry-grade
deployments can swap this for MinHash + LSH at scale; the
``rapidfuzz`` library would also work as a near-drop-in.

Two helpers:

* :func:`dedupe_events` — collapse near-duplicate events; returns
  the survivor list.
* :func:`fingerprint` — stable lower-case token-set hash used for
  fast pre-filtering.
"""
from __future__ import annotations

import hashlib
import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default similarity threshold. 0.85 = "almost the same sentence";
# 0.95 = "exact paraphrase". Tunable per call.
DEFAULT_DEDUP_THRESHOLD: float = 0.85

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalise(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(_TOKEN_RE.findall(text.lower()))


def fingerprint(text: Optional[str]) -> str:
    """Stable SHA-256 over lower-case alpha-numeric tokens.

    Two events with the same fingerprint have identical token sets —
    safe to treat as duplicates without measuring similarity.
    """
    payload = _normalise(text).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def dedupe_events(
    events: List[Dict],
    *,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    content_key: str = "content",
) -> List[Dict]:
    """Drop near-duplicate events from ``events``.

    First pass: bucket by fingerprint — events with identical token
    sets collapse to the first occurrence (cheap, O(N)).

    Second pass: O(N²) pairwise SequenceMatcher comparison within
    each bucket-of-different-fingerprint set. For typical TCMGM
    batches (≤ 100 events) the absolute cost stays ≤ 50 ms.

    Args:
        events: List of dicts with at least ``content_key``.
        threshold: Similarity floor in ``[0, 1]``. Higher = stricter.
        content_key: Dict key holding the comparable text.

    Returns:
        List of survivors in original order. Each survivor gains a
        ``"dedup_collapsed"`` int counting how many duplicates it
        absorbed (0 when unique).
    """
    if not events:
        return []

    survivors: List[Dict] = []
    survivor_norms: List[str] = []
    fingerprints_seen: Dict[str, int] = {}

    for event in events:
        text = event.get(content_key, "")
        if not isinstance(text, str):
            text = str(text or "")
        fp = fingerprint(text)

        # Fast path — identical token set.
        if fp in fingerprints_seen:
            idx = fingerprints_seen[fp]
            survivors[idx]["dedup_collapsed"] = (
                survivors[idx].get("dedup_collapsed", 0) + 1
            )
            continue

        # Slow path — pairwise similarity against existing survivors.
        norm = _normalise(text)
        duplicate_idx = -1
        for i, prior in enumerate(survivor_norms):
            if _ratio(norm, prior) >= threshold:
                duplicate_idx = i
                break
        if duplicate_idx >= 0:
            survivors[duplicate_idx]["dedup_collapsed"] = (
                survivors[duplicate_idx].get("dedup_collapsed", 0) + 1
            )
            continue

        # Survivor.
        new_event = dict(event)
        new_event.setdefault("dedup_collapsed", 0)
        survivors.append(new_event)
        survivor_norms.append(norm)
        fingerprints_seen[fp] = len(survivors) - 1

    if len(survivors) != len(events):
        logger.info(
            "dedupe_events: %d → %d (collapsed %d duplicates)",
            len(events),
            len(survivors),
            len(events) - len(survivors),
        )
    return survivors


__all__ = ["dedupe_events", "fingerprint", "DEFAULT_DEDUP_THRESHOLD"]
