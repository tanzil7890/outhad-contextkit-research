"""Phase 6 — ``GraphFirstRetriever`` seed→expand→re-rank behaviour.

The retriever is exercised against a real :class:`ContextGraph` + NetworkX
backend so the BFS + scoring logic is tested end-to-end. No vector store,
embedding model, or LLM is required.
"""
from __future__ import annotations

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.context_graph.retriever import GraphFirstRetriever
from outhad_contextkit.memory.context_graph.types import EdgeType


def _make_cg(**overrides):
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return build_context_graph(cfg), cfg


def _retriever(cg, cfg):
    return GraphFirstRetriever(cg, cfg.retrieval)


def test_empty_seed_returns_empty():
    cg, cfg = _make_cg()
    out = _retriever(cg, cfg).retrieve("anything", [], limit=5)
    assert out == {"results": [], "subgraph": []}


def test_seed_only_returns_seed_ordered_by_score():
    cg, cfg = _make_cg()
    cg.upsert_memory_node("a", "alpha", user_id="u1")
    cg.upsert_memory_node("b", "beta", user_id="u1")
    seeds = [
        {"id": "a", "memory": "alpha", "score": 0.8},
        {"id": "b", "memory": "beta", "score": 0.4},
    ]
    out = _retriever(cg, cfg).retrieve("alpha", seeds, limit=5)
    ids = [r["id"] for r in out["results"]]
    assert ids[0] == "a"  # higher dense + lexical "alpha"
    assert set(ids) == {"a", "b"}
    assert out["subgraph"] == []  # no edges traversed


def test_two_hop_neighbour_surfaces_via_graph_expansion():
    """Proves graph expansion does real work: a node with zero dense score
    but strong edge connectivity to a seed shows up in the result set."""
    cg, cfg = _make_cg()
    cfg.retrieval.expansion_depth = 2
    cfg.retrieval.edge_weight_floor = 0.1
    cfg.retrieval.max_candidates = 20
    for mid in ("seed", "hop1", "hop2"):
        cg.upsert_memory_node(mid, mid, user_id="u1")
    cg.upsert_edge("seed", "hop1", EdgeType.REPLY_TO, weight=0.9)
    cg.upsert_edge("hop1", "hop2", EdgeType.TOPIC_SIMILAR, weight=0.8)

    seeds = [{"id": "seed", "memory": "seed", "score": 0.9}]
    out = _retriever(cg, cfg).retrieve("seed", seeds, limit=10)
    returned = {r["id"] for r in out["results"]}
    assert "seed" in returned
    assert "hop1" in returned
    assert "hop2" in returned
    # hop2's contribution must come from graph reach, not dense/lexical.
    hop2 = next(r for r in out["results"] if r["id"] == "hop2")
    assert hop2["context_graph"]["graph"] > 0.0
    assert hop2["context_graph"]["was_seed"] is False
    assert len(out["subgraph"]) >= 2


def test_expansion_respects_edge_weight_floor():
    cg, cfg = _make_cg()
    cfg.retrieval.expansion_depth = 2
    cfg.retrieval.edge_weight_floor = 0.5  # cut low-weight edges
    cg.upsert_memory_node("seed", "seed", user_id="u1")
    cg.upsert_memory_node("weak", "weak", user_id="u1")
    cg.upsert_edge("seed", "weak", EdgeType.TOPIC_SIMILAR, weight=0.2)

    seeds = [{"id": "seed", "memory": "seed", "score": 0.9}]
    out = _retriever(cg, cfg).retrieve("seed", seeds, limit=10)
    ids = {r["id"] for r in out["results"]}
    assert "weak" not in ids


def test_archived_nodes_excluded_by_default():
    cg, cfg = _make_cg()
    cfg.retrieval.expansion_depth = 1
    cfg.retrieval.edge_weight_floor = 0.0
    cg.upsert_memory_node("seed", "seed", user_id="u1")
    cg.upsert_memory_node("stale", "stale", user_id="u1")
    cg.upsert_edge("seed", "stale", EdgeType.TEMPORAL_NEXT, weight=0.9)
    cg.archive_memory_node("stale")

    seeds = [{"id": "seed", "memory": "seed", "score": 0.9}]
    out = _retriever(cg, cfg).retrieve("seed", seeds, limit=10)
    ids = {r["id"] for r in out["results"]}
    assert "stale" not in ids

    # Now opt in.
    out2 = _retriever(cg, cfg).retrieve(
        "seed", seeds, limit=10, include_archived=True
    )
    ids2 = {r["id"] for r in out2["results"]}
    assert "stale" in ids2


def test_payload_resolver_supplies_missing_content():
    cg, cfg = _make_cg()
    cfg.retrieval.expansion_depth = 1
    cg.upsert_memory_node("seed", "seed", user_id="u1")
    cg.upsert_memory_node("far", "", user_id="u1")  # empty text intentionally
    cg.upsert_edge("seed", "far", EdgeType.DOCUMENT_LINK, weight=0.7)

    resolver_calls = {}

    def resolver(ids):
        for i in ids:
            resolver_calls[i] = True
        return {"far": {"id": "far", "memory": "faraway content", "score": 0.0}}

    seeds = [{"id": "seed", "memory": "seed", "score": 0.9}]
    out = _retriever(cg, cfg).retrieve(
        "faraway", seeds, limit=10, payload_resolver=resolver
    )
    assert "far" in resolver_calls
    far_entry = next(r for r in out["results"] if r["id"] == "far")
    assert far_entry["memory"] == "faraway content"


def test_max_candidates_caps_expansion():
    cg, cfg = _make_cg()
    cfg.retrieval.expansion_depth = 3
    cfg.retrieval.edge_weight_floor = 0.0
    cfg.retrieval.max_candidates = 3  # seed + 2 more
    cg.upsert_memory_node("seed", "seed", user_id="u1")
    for i in range(10):
        cg.upsert_memory_node(f"n{i}", f"n{i}", user_id="u1")
        cg.upsert_edge("seed", f"n{i}", EdgeType.TOPIC_SIMILAR, weight=0.9)

    seeds = [{"id": "seed", "memory": "seed", "score": 0.9}]
    out = _retriever(cg, cfg).retrieve("seed", seeds, limit=50)
    assert len(out["results"]) <= 3


def test_subgraph_is_deduplicated():
    cg, cfg = _make_cg()
    cfg.retrieval.expansion_depth = 2
    cfg.retrieval.edge_weight_floor = 0.1
    cg.upsert_memory_node("a", "a", user_id="u1")
    cg.upsert_memory_node("b", "b", user_id="u1")
    cg.upsert_memory_node("c", "c", user_id="u1")
    cg.upsert_edge("a", "b", EdgeType.REPLY_TO, weight=0.9)
    cg.upsert_edge("b", "c", EdgeType.REPLY_TO, weight=0.9)

    seeds = [{"id": "a", "memory": "a", "score": 0.9}]
    out = _retriever(cg, cfg).retrieve("a", seeds, limit=10)
    sigs = {(e["src"], e["dst"], e["type"]) for e in out["subgraph"]}
    assert sigs == {("a", "b", "REPLY_TO"), ("b", "c", "REPLY_TO")}
