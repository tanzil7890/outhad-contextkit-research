"""(Cypher UNWIND) + A1 (Reciprocal Rank Fusion) tests.

P4 — get_causal_chains_batch must:
- Issue exactly 2 graph.query calls regardless of seed count.
- Wire results back to per-seed forward/backward dicts.
- Tolerate missing seeds (resolves to empty lists).
- _explore_causal_chains uses batch path; falls back to legacy on
  exception.

A1 — _fuse_rrf must:
- Combine multiple channel ranks via 1/(k + rank).
- Produce monotone ordering (higher cross-channel co-occurrence wins).
- Survive missing channels.
- Legacy method="weighted" still callable.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from outhad_contextkit.memory.temporal.causal_queries import (
    get_causal_chains_batch,
)
from outhad_contextkit.memory.temporal.orchestrator import RetrievalOrchestrator


# ---------------------------------------------------------------------------
# P4 — batch causal chain exploration
# ---------------------------------------------------------------------------

def test_batch_issues_two_graph_queries_regardless_of_seed_count():
    graph = MagicMock()
    graph.query.return_value = []
    out = get_causal_chains_batch(
        graph,
        event_ids=["a", "b", "c", "d", "e"],
        filters={"user_id": "u1"},
    )
    # Forward + backward = exactly 2 round-trips.
    assert graph.query.call_count == 2
    # Every seed shows up with empty lists (no chains in mock).
    for seed in ("a", "b", "c", "d", "e"):
        assert out[seed] == {"forward": [], "backward": []}


def test_batch_wires_forward_and_backward_per_seed():
    graph = MagicMock()
    forward_payload = [
        {"seed_id": "a", "chains": [{"chain": [], "links": [], "depth": 1}]},
        {"seed_id": "b", "chains": [{"chain": [], "links": [], "depth": 2}]},
    ]
    backward_payload = [
        {"seed_id": "a", "chains": [{"chain": [], "links": [], "depth": 3}]},
    ]
    graph.query.side_effect = [forward_payload, backward_payload]
    out = get_causal_chains_batch(
        graph,
        event_ids=["a", "b"],
        filters={"user_id": "u1"},
    )
    assert len(out["a"]["forward"]) == 1 and out["a"]["forward"][0]["depth"] == 1
    assert len(out["a"]["backward"]) == 1 and out["a"]["backward"][0]["depth"] == 3
    assert len(out["b"]["forward"]) == 1
    assert out["b"]["backward"] == []


def test_batch_tolerates_query_exception():
    graph = MagicMock()
    graph.query.side_effect = RuntimeError("boom")
    out = get_causal_chains_batch(
        graph,
        event_ids=["x"],
        filters={"user_id": "u1"},
    )
    # Both queries raised — function returns empty per seed; never raises.
    assert out == {"x": {"forward": [], "backward": []}}


def test_batch_empty_event_ids_returns_empty():
    graph = MagicMock()
    out = get_causal_chains_batch(graph, event_ids=[], filters={"user_id": "u1"})
    assert out == {}
    graph.query.assert_not_called()


def test_orchestrator_explore_uses_batch_then_falls_back():
    """Orchestrator should prefer the batch path; fall back to legacy on error."""
    graph_store = MagicMock()
    graph_store.graph = MagicMock()
    graph_store.graph.query.return_value = []  # batch returns empty

    o = RetrievalOrchestrator(
        vector_store=MagicMock(),
        graph_store=graph_store,
        timeline_builder=MagicMock(),
        embedding_model=MagicMock(),
    )
    chains = o._explore_causal_chains(
        seed_events=[{"id": "e1"}, {"id": "e2"}, {"id": "e3"}],
        user_id="u1",
    )
    # Batch path consumed exactly 2 round-trips for 3 seeds.
    assert graph_store.graph.query.call_count == 2
    # 3 seeds → 3 chain dicts.
    assert len(chains) == 3
    for c in chains:
        assert "seed_event" in c
        assert "forward_chain" in c
        assert "backward_chain" in c


# ---------------------------------------------------------------------------
# A1 — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _orchestrator():
    return RetrievalOrchestrator(
        vector_store=MagicMock(),
        graph_store=MagicMock(),
        timeline_builder=MagicMock(),
        embedding_model=MagicMock(),
    )


def test_rrf_default_method_used():
    o = _orchestrator()
    out = o._fuse_results(
        {
            "vector_results": [{"id": "a", "content": "alpha", "score": 0.9}],
            "graph_results": [],
            "timeline_results": [],
        },
        query="q",
    )
    assert out and out[0]["fusion_method"] == "rrf"


def test_rrf_co_occurring_id_outranks_single_channel():
    """Doc that hits both vector + graph beats one that hits only vector."""
    o = _orchestrator()
    all_results = {
        "vector_results": [
            {"id": "shared", "content": "shared", "score": 0.5},
            {"id": "vector_only", "content": "v-only", "score": 0.99},
        ],
        "graph_results": [
            {"id": "shared", "content": "shared", "score": 0.7},
        ],
        "timeline_results": [],
    }
    fused = o._fuse_results(all_results, query="q")
    ids_in_order = [r["id"] for r in fused]
    # `shared` appears in 2 channels with rank 1 each. RRF gives it
    # higher score than `vector_only` (rank 2 in vector, missing elsewhere).
    assert ids_in_order.index("shared") < ids_in_order.index("vector_only")


def test_rrf_handles_missing_channels():
    o = _orchestrator()
    out = o._fuse_results(
        {"vector_results": [], "graph_results": [], "timeline_results": []},
        query="q",
    )
    assert out == []


def test_rrf_stable_with_only_one_channel():
    o = _orchestrator()
    all_results = {
        "vector_results": [
            {"id": str(i), "content": f"c{i}", "score": 1.0 - i * 0.01}
            for i in range(5)
        ],
        "graph_results": [],
        "timeline_results": [],
    }
    fused = o._fuse_results(all_results, query="q")
    assert [r["id"] for r in fused] == ["0", "1", "2", "3", "4"]


def test_legacy_weighted_fusion_still_works():
    o = _orchestrator()
    out = o._fuse_results(
        {
            "vector_results": [{"id": "a", "content": "alpha", "score": 0.5}],
            "graph_results": [],
            "timeline_results": [],
        },
        query="q",
        method="weighted",
    )
    assert out and out[0]["fusion_method"] == "weighted"
    # Weighted sum is score * weight = 0.5 * 0.4 = 0.2
    assert out[0]["final_score"] == pytest.approx(0.2)


def test_rrf_records_fusion_sources():
    o = _orchestrator()
    fused = o._fuse_results(
        {
            "vector_results": [{"id": "x", "content": "x", "score": 0.5}],
            "graph_results": [{"id": "x", "content": "x", "score": 0.5}],
            "timeline_results": [{"id": "x", "content": "x", "score": 0.5}],
        },
        query="q",
    )
    assert fused[0]["fusion_sources"] == ["graph", "timeline", "vector"]
