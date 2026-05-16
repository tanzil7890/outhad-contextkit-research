""" ``access_count`` tracking + ε·frequency scoring term.

Covers:
* ``bump_access_count`` primitive on the default NetworkX backend.
* ``ContextGraph.record_access`` facade wrapper (opt-in, guarded).
* ``normalised_frequency_with_peak`` log1p normalisation.
* ``_rerank_candidates`` parity: ``epsilon_frequency = 0`` → pre-F2 ranking.
* ε > 0 with populated counts → frequent memory gets boosted.
"""
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
from outhad_contextkit.memory.context_graph.scoring import (
    normalised_frequency,
    normalised_frequency_with_peak,
)
from outhad_contextkit.memory.context_graph.types import (
    EdgeType,
    MemoryEdge,
    MemoryNode,
)


def _now() -> datetime:
    return datetime(2026, 4, 24, 10, 0, 0)


def _backend_with_two_nodes() -> NetworkXBackend:
    backend = NetworkXBackend()
    backend.upsert_node(
        MemoryNode(id="m1", hash="h1", created_at=_now(), updated_at=_now())
    )
    backend.upsert_node(
        MemoryNode(id="m2", hash="h2", created_at=_now(), updated_at=_now())
    )
    return backend


# ---------------------------------------------------------------------------
#  Backend primitive
# ---------------------------------------------------------------------------


def test_bump_access_count_increments_and_updates_timestamp():
    backend = _backend_with_two_nodes()
    first = backend.bump_access_count("m1")
    second = backend.bump_access_count("m1", delta=2)
    assert first == 1
    assert second == 3
    node = backend.get_node("m1")
    assert node is not None
    assert node.access_count == 3
    assert node.last_accessed_at is not None


def test_bump_access_count_returns_none_for_unknown_node():
    backend = _backend_with_two_nodes()
    assert backend.bump_access_count("missing") is None


def test_bump_access_count_clamps_at_zero():
    backend = _backend_with_two_nodes()
    backend.bump_access_count("m1", delta=1)
    result = backend.bump_access_count("m1", delta=-10)
    assert result == 0
    node = backend.get_node("m1")
    assert node.access_count == 0


def test_max_access_count_tracks_peak():
    backend = _backend_with_two_nodes()
    backend.bump_access_count("m1", delta=5)
    backend.bump_access_count("m2", delta=2)
    assert backend.max_access_count() == 5


# ---------------------------------------------------------------------------
#  Facade wrapper
# ---------------------------------------------------------------------------


def test_record_access_returns_none_when_disabled():
    backend = _backend_with_two_nodes()
    cg = ContextGraph(
        config=ContextGraphConfig(enabled=False),
        backend=backend,
        changelog=None,
    )
    # Disabled → facade short-circuits, counter stays 0.
    assert cg.record_access("m1") is None
    assert backend.get_node("m1").access_count == 0


def test_record_access_enabled_emits_and_increments():
    backend = _backend_with_two_nodes()
    cg = ContextGraph(
        config=ContextGraphConfig(enabled=True, log_changes=False),
        backend=backend,
        changelog=None,
    )
    count = cg.record_access("m1")
    assert count == 1
    assert cg.max_access_count() == 1


# ---------------------------------------------------------------------------
#  Normalisation
# ---------------------------------------------------------------------------


def test_normalised_frequency_zero_when_count_is_zero():
    node = MemoryNode(
        id="m", hash="h", created_at=_now(), updated_at=_now(), access_count=0
    )
    assert normalised_frequency_with_peak(node, peak=10) == 0.0


def test_normalised_frequency_monotonic():
    lo = MemoryNode(
        id="lo", hash="h", created_at=_now(), updated_at=_now(), access_count=1
    )
    hi = MemoryNode(
        id="hi", hash="h", created_at=_now(), updated_at=_now(), access_count=10
    )
    assert normalised_frequency_with_peak(hi, peak=10) > normalised_frequency_with_peak(
        lo, peak=10
    )


def test_normalised_frequency_at_peak_equals_one():
    node = MemoryNode(
        id="m", hash="h", created_at=_now(), updated_at=_now(), access_count=8
    )
    assert normalised_frequency_with_peak(node, peak=8) == pytest.approx(1.0)


def test_normalised_frequency_reads_backend_peak_when_present():
    backend = _backend_with_two_nodes()
    node = backend.get_node("m1")
    node.access_count = 3
    backend.upsert_node(node)
    cg = ContextGraph(
        config=ContextGraphConfig(enabled=True, log_changes=False),
        backend=backend,
        changelog=None,
    )
    score = normalised_frequency(backend.get_node("m1"), cg)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
#  Rerank parity + boost
# ---------------------------------------------------------------------------


def _seed_graph() -> ContextGraph:
    backend = NetworkXBackend()
    backend.upsert_node(
        MemoryNode(id="hot", hash="h", created_at=_now(), updated_at=_now())
    )
    backend.upsert_node(
        MemoryNode(id="cold", hash="h", created_at=_now(), updated_at=_now())
    )
    return ContextGraph(
        config=ContextGraphConfig(enabled=True, log_changes=False),
        backend=backend,
        changelog=None,
    )


def _rank_ids(results):
    return [r["id"] for r in results["results"]]


def test_rerank_identical_when_epsilon_zero():
    cg = _seed_graph()
    # Equal dense scores, no edges, identical lexical — order determined by
    # seed_top_k trim which is stable by dense score then insertion.
    seeds = [
        {"id": "cold", "memory": "cold text", "score": 0.6},
        {"id": "hot", "memory": "hot text", "score": 0.6},
    ]
    cfg_off = RetrievalConfig(
        alpha_dense=1.0,
        beta_bm25=0.0,
        gamma_graph=0.0,
        epsilon_frequency=0.0,
        expansion_depth=0,
    )
    retriever_off = GraphFirstRetriever(cg, cfg_off)
    baseline = _rank_ids(retriever_off.retrieve("query", seeds, limit=2))

    # Pump 'hot' with access but leave epsilon at 0 — order must not shift.
    cg.backend.bump_access_count("hot", delta=20)
    boosted = _rank_ids(retriever_off.retrieve("query", seeds, limit=2))
    assert baseline == boosted


def test_rerank_boosts_frequent_memory_when_epsilon_positive():
    cg = _seed_graph()
    cg.backend.bump_access_count("hot", delta=20)
    seeds = [
        {"id": "cold", "memory": "cold text", "score": 0.6},
        {"id": "hot", "memory": "hot text", "score": 0.6},
    ]
    cfg_on = RetrievalConfig(
        alpha_dense=0.5,
        beta_bm25=0.0,
        gamma_graph=0.0,
        epsilon_frequency=0.5,
        expansion_depth=0,
    )
    retriever_on = GraphFirstRetriever(cg, cfg_on)
    ranked = retriever_on.retrieve("query", seeds, limit=2)
    ids = [r["id"] for r in ranked["results"]]
    assert ids[0] == "hot"
    # Frequency contribution is surfaced in the debug payload.
    assert ranked["results"][0]["context_graph"]["frequency"] > 0.0


def test_edge_weight_from_relevance_still_applies_with_frequency():
    """High access_count must not overpower a near-zero relevance node."""
    cg = _seed_graph()
    cg.backend.bump_access_count("hot", delta=20)
    node = cg.backend.get_node("hot")
    node.relevance = 0.0
    cg.backend.upsert_node(node)
    seeds = [
        {"id": "cold", "memory": "cold text", "score": 0.3},
        {"id": "hot", "memory": "hot text", "score": 0.3},
    ]
    cfg_on = RetrievalConfig(
        alpha_dense=0.5,
        beta_bm25=0.0,
        gamma_graph=0.0,
        epsilon_frequency=0.5,
        expansion_depth=0,
    )
    retriever_on = GraphFirstRetriever(cg, cfg_on)
    ranked = retriever_on.retrieve("query", seeds, limit=2)
    assert ranked["results"][0]["id"] == "cold"
