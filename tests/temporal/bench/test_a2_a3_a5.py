""" (JSON-schema causal prompt) + A3 (event dedup) +
A5 (timeline + semantic rerank) tests.

A2 — confidence threshold + few-shot prompt:
- Low-confidence (< 0.6) edges dropped by extractor.
- Self-loop edges (cause==effect) dropped.
- Confidence values clamped to [0, 1] when LLM returns out-of-range.
- New prompt body still produces parseable JSON when LLM mocked.

A3 — dedupe_events:
- Identical events collapse to one survivor with `dedup_collapsed`.
- Near-duplicates above threshold collapse.
- Below-threshold pairs preserved as distinct.
- Empty input returns empty list.
- Fingerprint stable across whitespace + case differences.

A5 — semantic timeline rerank:
- Events with higher token overlap to query rank above older events.
- Rerank only fires when query supplied + embedding model present.
- Rerank failure does not break the search (returns legacy list).
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from outhad_contextkit.memory.temporal.causal_extractor import (
    CAUSAL_MIN_CONFIDENCE,
    CausalExtractor,
    clear_causal_llm_cache,
)
from outhad_contextkit.memory.temporal.dedup import (
    DEFAULT_DEDUP_THRESHOLD,
    dedupe_events,
    fingerprint,
)
from outhad_contextkit.memory.temporal.orchestrator import RetrievalOrchestrator
from outhad_contextkit.memory.temporal.types import TimeWindow


# ---------------------------------------------------------------------------
# A2 — JSON-schema causal prompt + confidence threshold
# ---------------------------------------------------------------------------

def _llm_returning(payload: str) -> MagicMock:
    llm = MagicMock()
    llm.generate_response.return_value = payload
    return llm


def test_low_confidence_edges_dropped():
    clear_causal_llm_cache()
    payload = (
        '{"causal_links": ['
        '{"cause_id": "e1", "effect_id": "e2", "causal_type": "leads_to", '
        '"confidence": 0.3, "evidence": "weak"},'
        '{"cause_id": "e1", "effect_id": "e3", "causal_type": "leads_to", '
        '"confidence": 0.8, "evidence": "strong"}'
        ']}'
    )
    ex = CausalExtractor(llm=_llm_returning(payload))
    events = [
        {"id": "e1", "timestamp": "t1", "content": "A"},
        {"id": "e2", "timestamp": "t2", "content": "B"},
        {"id": "e3", "timestamp": "t3", "content": "C"},
    ]
    links = ex.extract_causal_links(events, use_llm=True)
    assert len(links) == 1
    assert links[0].effect_id == "e3"
    assert links[0].confidence == pytest.approx(0.8)


def test_self_loop_edges_dropped():
    clear_causal_llm_cache()
    payload = (
        '{"causal_links": ['
        '{"cause_id": "e1", "effect_id": "e1", "causal_type": "leads_to", '
        '"confidence": 0.95, "evidence": "self-loop"}'
        ']}'
    )
    ex = CausalExtractor(llm=_llm_returning(payload))
    events = [{"id": "e1", "timestamp": "t1", "content": "A"}]
    assert ex.extract_causal_links(events, use_llm=True) == []


def test_confidence_clamped_above_one():
    clear_causal_llm_cache()
    payload = (
        '{"causal_links": ['
        '{"cause_id": "e1", "effect_id": "e2", "causal_type": "leads_to", '
        '"confidence": 99, "evidence": "out of range"}'
        ']}'
    )
    ex = CausalExtractor(llm=_llm_returning(payload))
    events = [
        {"id": "e1", "timestamp": "t1", "content": "A"},
        {"id": "e2", "timestamp": "t2", "content": "B"},
    ]
    links = ex.extract_causal_links(events, use_llm=True)
    assert len(links) == 1
    assert links[0].confidence == pytest.approx(1.0)


def test_min_confidence_constant_exposed():
    assert 0.0 <= CAUSAL_MIN_CONFIDENCE <= 1.0


# ---------------------------------------------------------------------------
# A3 — dedupe_events
# ---------------------------------------------------------------------------

def test_dedupe_collapses_identical_content():
    events = [
        {"id": "e1", "content": "Alice loves pizza"},
        {"id": "e2", "content": "Alice loves pizza"},
        {"id": "e3", "content": "Alice loves pizza"},
    ]
    out = dedupe_events(events)
    assert len(out) == 1
    assert out[0]["dedup_collapsed"] == 2


def test_dedupe_collapses_near_duplicates():
    events = [
        {"id": "e1", "content": "Alice loves pizza"},
        {"id": "e2", "content": "alice  loves   pizza."},  # whitespace + case + punct
    ]
    out = dedupe_events(events, threshold=0.85)
    assert len(out) == 1


def test_dedupe_keeps_distinct_events():
    events = [
        {"id": "e1", "content": "Alice loves pizza"},
        {"id": "e2", "content": "Bob hates broccoli"},
    ]
    out = dedupe_events(events)
    assert len(out) == 2


def test_dedupe_empty_input():
    assert dedupe_events([]) == []


def test_fingerprint_stable_across_whitespace_and_case():
    a = fingerprint("  Hello World ")
    b = fingerprint("hello\tworld")
    assert a == b


def test_fingerprint_changes_on_real_change():
    assert fingerprint("Alice") != fingerprint("Bob")


def test_default_threshold_in_unit_interval():
    assert 0.5 <= DEFAULT_DEDUP_THRESHOLD <= 1.0


def test_extractor_runs_dedup_by_default():
    """CausalExtractor.extract_causal_links collapses dupes before LLM."""
    clear_causal_llm_cache()
    payload = '{"causal_links": []}'
    llm = _llm_returning(payload)
    ex = CausalExtractor(llm=llm)
    events = [
        {"id": "e1", "timestamp": "t", "content": "Alice loves pizza"},
        {"id": "e2", "timestamp": "t", "content": "Alice loves pizza"},
    ]
    ex.extract_causal_links(events, use_llm=True)
    # Prompt body should contain only one event line.
    prompt = llm.generate_response.call_args[1]["messages"][0]["content"]
    assert prompt.count("Alice loves pizza") == 1


# ---------------------------------------------------------------------------
# A5 — Timeline semantic rerank
# ---------------------------------------------------------------------------

def _orch_with_timeline_events(events):
    timeline_builder = MagicMock()
    timeline_builder.get_timeline.return_value = events
    embedding_model = MagicMock()
    embedding_model.embed.return_value = [0.1] * 8  # any vector — token fallback used
    return RetrievalOrchestrator(
        vector_store=MagicMock(),
        graph_store=MagicMock(),
        timeline_builder=timeline_builder,
        embedding_model=embedding_model,
    )


def test_timeline_rerank_promotes_high_overlap_events():
    o = _orch_with_timeline_events([
        {"id": "old1", "content": "weather report yesterday", "timestamp": "t1"},
        {"id": "match", "content": "deployed new API and saw latency rise", "timestamp": "t2"},
        {"id": "old2", "content": "lunch was good", "timestamp": "t3"},
    ])
    out = o._timeline_search(
        user_id="u1",
        time_window=TimeWindow(
            start=datetime(2024, 1, 1), end=datetime(2024, 12, 31)
        ),
        query="deployed API latency",
    )
    ids = [r["id"] for r in out]
    # Rerank should put the matching event first.
    assert ids[0] == "match"


def test_timeline_rerank_skipped_without_query():
    """No query → legacy time order preserved (via stable sort behaviour)."""
    events = [
        {"id": "a", "content": "alpha", "timestamp": "t1"},
        {"id": "b", "content": "beta", "timestamp": "t2"},
    ]
    o = _orch_with_timeline_events(events)
    out = o._timeline_search(
        user_id="u1",
        time_window=TimeWindow(
            start=datetime(2024, 1, 1), end=datetime(2024, 12, 31)
        ),
        query=None,
    )
    # Without rerank, scores are uniform 0.7 → original order kept.
    assert [r["id"] for r in out] == ["a", "b"]
    for r in out:
        assert r["score"] == 0.7


def test_timeline_rerank_failure_falls_back():
    """Any exception in rerank returns the legacy formatted list."""
    o = _orch_with_timeline_events([
        {"id": "x", "content": "hi", "timestamp": "t1"},
    ])
    o.embedding_model = MagicMock()
    o.embedding_model.embed.side_effect = RuntimeError("boom")
    out = o._timeline_search(
        user_id="u1",
        time_window=TimeWindow(
            start=datetime(2024, 1, 1), end=datetime(2024, 12, 31)
        ),
        query="anything",
    )
    # Search must still return something (legacy shape).
    assert len(out) == 1
    assert out[0]["id"] == "x"
