""" QuerySuccessStore SQLite integration + SuccessProvider.

Covers:
* Schema bootstrap is idempotent.
* record → aggregate → hit_rate round-trip.
* hit_rate returns None when total < min_samples; full ratio otherwise.
* User scope filtering.
* Window filter (time-based) drops older rows.
* top_memories returns memories ranked by hit_rate.
* reset clears rows by user / by query_hash.
* Durability across instances.
* SuccessProvider: cache hit, None query_hash → 0.0, broken store → 0.0.
* Retriever ζ·success integration: positive hit_rate boosts, parity at ζ=0.
* record_feedback writes a success row when success.enabled=True.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph.backends.networkx_backend import (
    NetworkXBackend,
)
from outhad_contextkit.memory.context_graph.config import (
    ContextGraphConfig,
    RetrievalConfig,
)
from outhad_contextkit.memory.context_graph.facade import ContextGraph
from outhad_contextkit.memory.context_graph.retriever import GraphFirstRetriever
from outhad_contextkit.memory.context_graph.types import MemoryNode
from outhad_contextkit.memory.personalized.success_store import (
    QuerySuccessStore,
    SuccessProvider,
)


def _now():
    return datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# QuerySuccessStore
# ---------------------------------------------------------------------------

def test_schema_bootstrap_is_idempotent(tmp_path):
    path = str(tmp_path / "qs.db")
    QuerySuccessStore(path)
    QuerySuccessStore(path)  # second open must not raise
    store = QuerySuccessStore(path)
    assert store.aggregate(query_hash="x", memory_id="m1") == {
        "helpful": 0,
        "unhelpful": 0,
        "total": 0,
    }


def test_record_returns_rowid(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    rid = store.record(query_hash="qh1", memory_id="m1", helpful=True, user_id="u1")
    assert rid > 0


def test_record_rejects_empty_keys(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    assert store.record(query_hash="", memory_id="m1", helpful=True) == -1
    assert store.record(query_hash="qh1", memory_id="", helpful=True) == -1


def test_aggregate_counts_helpful_and_total(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    store.record(query_hash="qh", memory_id="m1", helpful=True)
    store.record(query_hash="qh", memory_id="m1", helpful=True)
    store.record(query_hash="qh", memory_id="m1", helpful=False)
    stats = store.aggregate(query_hash="qh", memory_id="m1")
    assert stats == {"helpful": 2, "unhelpful": 1, "total": 3}


def test_hit_rate_requires_min_samples(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    store.record(query_hash="qh", memory_id="m1", helpful=True)
    store.record(query_hash="qh", memory_id="m1", helpful=True)
    # min_samples=3 but we only have 2 verdicts → None.
    assert store.hit_rate("qh", "m1", min_samples=3) is None
    store.record(query_hash="qh", memory_id="m1", helpful=False)
    rate = store.hit_rate("qh", "m1", min_samples=3)
    assert rate == pytest.approx(2 / 3)


def test_hit_rate_returns_none_for_empty_keys(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    assert store.hit_rate("", "m1") is None
    assert store.hit_rate("qh", "") is None


def test_user_scope_filters_aggregate(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    store.record(query_hash="qh", memory_id="m1", helpful=True, user_id="u1")
    store.record(query_hash="qh", memory_id="m1", helpful=False, user_id="u2")
    assert store.aggregate(query_hash="qh", memory_id="m1", user_id="u1")["total"] == 1
    assert (
        store.aggregate(query_hash="qh", memory_id="m1", user_id="u1")["helpful"] == 1
    )
    assert (
        store.aggregate(query_hash="qh", memory_id="m1", user_id="u2")["helpful"] == 0
    )


def test_window_filter_drops_old_rows(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    old_ts = datetime.now(timezone.utc) - timedelta(days=60)
    store.record(query_hash="qh", memory_id="m1", helpful=True, timestamp=old_ts)
    store.record(query_hash="qh", memory_id="m1", helpful=True)
    stats = store.aggregate(query_hash="qh", memory_id="m1", window_days=30)
    assert stats["total"] == 1


def test_top_memories_ranked_by_hit_rate(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    # m1: 3/3 helpful; m2: 2/3 helpful; m3: 0/2 helpful.
    for _ in range(3):
        store.record(query_hash="qh", memory_id="m1", helpful=True)
    for h in (True, True, False):
        store.record(query_hash="qh", memory_id="m2", helpful=h)
    for _ in range(2):
        store.record(query_hash="qh", memory_id="m3", helpful=False)
    top = store.top_memories("qh")
    ids = [t[0] for t in top]
    assert ids[0] == "m1"
    assert ids[1] == "m2"
    assert "m3" in ids


def test_reset_by_user_id(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    store.record(query_hash="qh", memory_id="m1", helpful=True, user_id="u1")
    store.record(query_hash="qh", memory_id="m1", helpful=True, user_id="u2")
    removed = store.reset(user_id="u1")
    assert removed == 1
    assert (
        store.aggregate(query_hash="qh", memory_id="m1", user_id="u1")["total"] == 0
    )
    assert (
        store.aggregate(query_hash="qh", memory_id="m1", user_id="u2")["total"] == 1
    )


def test_durability_across_instances(tmp_path):
    path = str(tmp_path / "qs.db")
    QuerySuccessStore(path).record(query_hash="qh", memory_id="m1", helpful=True)
    reopened = QuerySuccessStore(path)
    assert reopened.aggregate(query_hash="qh", memory_id="m1")["total"] == 1


# ---------------------------------------------------------------------------
# SuccessProvider
# ---------------------------------------------------------------------------

def test_provider_caches_per_memory_id(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    for _ in range(3):
        store.record(query_hash="qh", memory_id="m1", helpful=True)
    p = SuccessProvider(store, query_hash="qh", min_samples=3)
    first = p.hit_rate("m1")
    assert first == pytest.approx(1.0)
    # Second call hits the cache — record more rows after caching, the
    # cached value should remain.
    store.record(query_hash="qh", memory_id="m1", helpful=False)
    assert p.hit_rate("m1") == pytest.approx(1.0)


def test_provider_returns_zero_when_query_hash_none():
    p = SuccessProvider(None, query_hash=None)
    assert p.hit_rate("anything") == 0.0


def test_provider_returns_zero_on_store_error():
    class _Broken:
        def hit_rate(self, *_a, **_kw):
            raise RuntimeError("boom")

    p = SuccessProvider(_Broken(), query_hash="qh")
    assert p.hit_rate("m1") == 0.0


def test_provider_returns_zero_when_under_sample(tmp_path):
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    store.record(query_hash="qh", memory_id="m1", helpful=True)
    p = SuccessProvider(store, query_hash="qh", min_samples=3)
    # Only 1 verdict → hit_rate returns None → provider returns 0.0.
    assert p.hit_rate("m1") == 0.0


# ---------------------------------------------------------------------------
# Retriever ζ·success integration
# ---------------------------------------------------------------------------

def _graph_with(backend, ids):
    for mid in ids:
        backend.upsert_node(
            MemoryNode(
                id=mid,
                hash="h",
                created_at=_now().replace(tzinfo=None),
                updated_at=_now().replace(tzinfo=None),
            )
        )
    return ContextGraph(
        config=ContextGraphConfig(enabled=True, log_changes=False),
        backend=backend,
        changelog=None,
    )


def test_zeta_zero_parity(tmp_path):
    """ζ=0 → identical ranking whether provider is supplied or not."""
    backend = NetworkXBackend()
    cg = _graph_with(backend, ["a", "b"])
    seeds = [
        {"id": "a", "memory": "alpha", "score": 0.6},
        {"id": "b", "memory": "beta", "score": 0.6},
    ]
    cfg = RetrievalConfig(
        alpha_dense=1.0,
        beta_bm25=0.0,
        gamma_graph=0.0,
        zeta_success=0.0,
        expansion_depth=0,
    )
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    for _ in range(5):
        store.record(query_hash="qh", memory_id="a", helpful=True)
    provider = SuccessProvider(store, query_hash="qh", min_samples=1)
    without = GraphFirstRetriever(cg, cfg).retrieve("q", seeds, limit=2)
    with_provider = GraphFirstRetriever(cg, cfg).retrieve(
        "q", seeds, limit=2, success_provider=provider
    )
    assert [r["id"] for r in without["results"]] == [
        r["id"] for r in with_provider["results"]
    ]


def test_zeta_positive_boosts_high_hit_rate(tmp_path):
    """ζ>0 + hit_rate>0 lifts the matching memory above its peers."""
    backend = NetworkXBackend()
    cg = _graph_with(backend, ["winner", "tie"])
    seeds = [
        {"id": "tie", "memory": "tie", "score": 0.6},
        {"id": "winner", "memory": "winner", "score": 0.6},
    ]
    cfg = RetrievalConfig(
        alpha_dense=0.4,
        beta_bm25=0.0,
        gamma_graph=0.0,
        zeta_success=0.5,
        expansion_depth=0,
    )
    store = QuerySuccessStore(str(tmp_path / "qs.db"))
    for _ in range(5):
        store.record(query_hash="qh", memory_id="winner", helpful=True)
    provider = SuccessProvider(store, query_hash="qh", min_samples=3)
    out = GraphFirstRetriever(cg, cfg).retrieve(
        "q", seeds, limit=2, success_provider=provider
    )
    assert out["results"][0]["id"] == "winner"
    assert out["results"][0]["context_graph"]["success"] == pytest.approx(1.0)


def test_success_payload_zero_when_provider_absent(tmp_path):
    """Default debug payload exposes success=0.0 even without a provider."""
    backend = NetworkXBackend()
    cg = _graph_with(backend, ["a"])
    seeds = [{"id": "a", "memory": "alpha", "score": 0.5}]
    cfg = RetrievalConfig(
        alpha_dense=1.0, beta_bm25=0.0, gamma_graph=0.0, expansion_depth=0
    )
    out = GraphFirstRetriever(cg, cfg).retrieve("q", seeds, limit=1)
    assert out["results"][0]["context_graph"]["success"] == 0.0


# ---------------------------------------------------------------------------
# Memory.record_feedback writes a success row when success.enabled=True
# ---------------------------------------------------------------------------

def test_record_feedback_writes_success_row(tmp_path):
    """End-to-end: a helpful verdict creates a row in query_success."""
    from outhad_contextkit.configs.base import MemoryConfig
    from outhad_contextkit.memory.context_graph import build_context_graph
    from outhad_contextkit.memory.context_graph.config import (
        ContextGraphConfig as CGC,
    )
    from outhad_contextkit.memory.main import Memory
    from outhad_contextkit.memory.personalized.feedback_store import FeedbackStore

    obj = object.__new__(Memory)
    cfg = MemoryConfig()
    cfg.context_graph = CGC(enabled=True, backend="networkx", log_changes=False)
    cfg.mspr.enabled = True
    cfg.mspr.feedback.enabled = True
    cfg.mspr.success.enabled = True
    obj.config = cfg
    obj._context_graph = build_context_graph(cfg.context_graph)
    obj._context_graph_builder = None
    obj._feedback_store = FeedbackStore(str(tmp_path / "fb.db"))
    obj._success_store = QuerySuccessStore(str(tmp_path / "qs.db"))
    obj._intent_router = None
    obj._role_policy = None
    obj._context_graph.backend.upsert_node(
        MemoryNode(
            id="m1",
            hash="h",
            created_at=_now().replace(tzinfo=None),
            updated_at=_now().replace(tzinfo=None),
        )
    )
    Memory.record_feedback(
        obj, "m1", helpful=True, user_id="u1", query="What is X?"
    )
    # Row should exist for the SHA-256[:16] hash of "what is x?".
    from outhad_contextkit.memory.personalized.feedback_store import hash_query

    qh = hash_query("What is X?")
    stats = obj._success_store.aggregate(query_hash=qh, memory_id="m1")
    assert stats["total"] == 1
    assert stats["helpful"] == 1
