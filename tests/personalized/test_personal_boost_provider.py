"""Phase F4 — PersonalBoostProvider caching + rerank integration."""
from __future__ import annotations

from datetime import datetime

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
from outhad_contextkit.memory.personalized.providers import PersonalBoostProvider


class _FakeStore:
    def __init__(self, mapping):
        self.mapping = dict(mapping)
        self.calls = 0

    def personal_boost(self, memory_id, *, user_id=None):
        self.calls += 1
        return float(self.mapping.get(memory_id, 0.0))


def _now():
    return datetime(2026, 4, 24, 12, 0, 0)


def _graph_with(backend, ids):
    for mid in ids:
        backend.upsert_node(
            MemoryNode(id=mid, hash="h", created_at=_now(), updated_at=_now())
        )
    return ContextGraph(
        config=ContextGraphConfig(enabled=True, log_changes=False),
        backend=backend,
        changelog=None,
    )


def test_provider_caches_per_memory_id():
    store = _FakeStore({"a": 0.7, "b": -0.4})
    p = PersonalBoostProvider(store, user_id="u1")
    assert p.boost_for("a") == pytest.approx(0.7)
    assert p.boost_for("a") == pytest.approx(0.7)
    assert store.calls == 1  # cache hit on second lookup


def test_provider_clamps_to_unit_interval():
    p = PersonalBoostProvider(_FakeStore({"x": 17.0}))
    assert p.boost_for("x") == 1.0
    p2 = PersonalBoostProvider(_FakeStore({"x": -17.0}))
    assert p2.boost_for("x") == -1.0


def test_provider_tolerates_backend_errors():
    class _Broken:
        def personal_boost(self, *_a, **_kw):
            raise RuntimeError("boom")

    p = PersonalBoostProvider(_Broken())
    assert p.boost_for("m") == 0.0


def test_rerank_includes_delta_personal_term():
    backend = NetworkXBackend()
    cg = _graph_with(backend, ["boost", "base"])
    seeds = [
        {"id": "base", "memory": "base", "score": 0.6},
        {"id": "boost", "memory": "boost", "score": 0.6},
    ]
    cfg = RetrievalConfig(
        alpha_dense=0.5,
        beta_bm25=0.0,
        gamma_graph=0.0,
        delta_personal=0.5,
        expansion_depth=0,
    )
    provider = PersonalBoostProvider(_FakeStore({"boost": 0.9}))
    results = GraphFirstRetriever(cg, cfg).retrieve(
        "q", seeds, limit=2, personal_boost=provider
    )
    assert results["results"][0]["id"] == "boost"
    assert results["results"][0]["context_graph"]["personal"] == pytest.approx(0.9)


def test_delta_zero_is_parity_with_pre_f4():
    backend = NetworkXBackend()
    cg = _graph_with(backend, ["a", "b"])
    seeds = [
        {"id": "a", "memory": "alpha", "score": 0.6},
        {"id": "b", "memory": "beta", "score": 0.6},
    ]
    cfg_off = RetrievalConfig(
        alpha_dense=1.0,
        beta_bm25=0.0,
        gamma_graph=0.0,
        delta_personal=0.0,
        expansion_depth=0,
    )
    provider = PersonalBoostProvider(_FakeStore({"a": 1.0, "b": -1.0}))
    # Identical ranking whether provider is present or not — δ=0 short-circuits.
    without = GraphFirstRetriever(cg, cfg_off).retrieve("q", seeds, limit=2)
    with_provider = GraphFirstRetriever(cg, cfg_off).retrieve(
        "q", seeds, limit=2, personal_boost=provider
    )
    assert [r["id"] for r in without["results"]] == [
        r["id"] for r in with_provider["results"]
    ]


def test_penalty_demotes_memory_when_delta_positive():
    backend = NetworkXBackend()
    cg = _graph_with(backend, ["hi", "lo"])
    seeds = [
        {"id": "hi", "memory": "high", "score": 0.6},
        {"id": "lo", "memory": "low", "score": 0.6},
    ]
    cfg = RetrievalConfig(
        alpha_dense=0.4,
        beta_bm25=0.0,
        gamma_graph=0.0,
        delta_personal=0.5,
        expansion_depth=0,
    )
    provider = PersonalBoostProvider(_FakeStore({"hi": -0.8, "lo": 0.3}))
    results = GraphFirstRetriever(cg, cfg).retrieve(
        "q", seeds, limit=2, personal_boost=provider
    )
    assert results["results"][0]["id"] == "lo"
