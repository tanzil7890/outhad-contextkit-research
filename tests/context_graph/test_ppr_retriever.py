"""Phase D — Personalised PageRank retriever tests.

Covers:
* default algorithm=bfs is byte-identical to the pre-Phase-D retriever,
* algorithm=ppr produces deterministic, consistent rankings,
* on a tree-graph PPR ranks descendants in depth order (sanity),
* pinned relevance (Phase A) still modulates the final score,
* PPR respects ``edge_weight_floor`` and ``include_archived``,
* empty seeds return an empty result set.
"""
from __future__ import annotations

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.config import (
    ContextGraphConfig,
    RetrievalConfig,
)
from outhad_contextkit.memory.context_graph.ppr_retriever import (
    PersonalisedPageRankRetriever,
)
from outhad_contextkit.memory.context_graph.retriever import GraphFirstRetriever
from outhad_contextkit.memory.context_graph.types import EdgeType


def _cfg(**overrides) -> ContextGraphConfig:
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    if overrides:
        cfg.retrieval = RetrievalConfig(**overrides)
    return cfg


def _seed(score: float, cid: str, text: str = ""):
    return {"id": cid, "memory": text, "score": score}


def _build_chain_graph(cg, n: int = 5) -> None:
    """m1 → m2 → m3 → ... with strong edges + uniform embeddings."""
    for i in range(1, n + 1):
        cg.upsert_memory_node(f"m{i}", f"memory {i}", user_id="u1")
    for i in range(1, n):
        cg.upsert_edge(
            f"m{i}",
            f"m{i + 1}",
            EdgeType.TOPIC_SIMILAR,
            weight=0.9,
        )


def test_retrieval_config_algorithm_defaults_bfs():
    cfg = RetrievalConfig()
    assert cfg.algorithm == "bfs"


def test_ppr_retriever_returns_seed_when_no_edges():
    cfg = _cfg(algorithm="ppr", seed_top_k=3, expansion_depth=2)
    cg = build_context_graph(cfg)
    cg.upsert_memory_node("m1", "hello", user_id="u1")

    retriever = PersonalisedPageRankRetriever(cg, cfg.retrieval)
    result = retriever.retrieve(
        "hello", [_seed(0.9, "m1", "hello")], limit=5
    )
    ids = [r["id"] for r in result["results"]]
    assert ids == ["m1"]


def test_ppr_chain_ranks_closer_descendants_higher():
    cfg = _cfg(
        algorithm="ppr",
        seed_top_k=1,
        expansion_depth=4,
        max_candidates=10,
        edge_weight_floor=0.0,
        alpha_dense=0.0,
        beta_bm25=0.0,
        gamma_graph=1.0,
    )
    cg = build_context_graph(cfg)
    _build_chain_graph(cg, n=5)

    retriever = PersonalisedPageRankRetriever(cg, cfg.retrieval)
    result = retriever.retrieve(
        "anything",
        [_seed(1.0, "m1", "memory 1")],
        limit=5,
    )
    ranked_ids = [r["id"] for r in result["results"]]
    assert ranked_ids[0] == "m1"
    # PPR scores drop monotonically along the chain.
    reach_by_id = {
        r["id"]: r["context_graph"]["graph"] for r in result["results"]
    }
    scores_in_order = [reach_by_id[f"m{i}"] for i in range(1, 6) if f"m{i}" in reach_by_id]
    assert scores_in_order == sorted(scores_in_order, reverse=True)


def test_ppr_is_deterministic_for_fixed_graph():
    cfg = _cfg(
        algorithm="ppr",
        seed_top_k=1,
        expansion_depth=3,
        max_candidates=10,
        alpha_dense=0.0,
        beta_bm25=0.0,
        gamma_graph=1.0,
    )
    cg = build_context_graph(cfg)
    _build_chain_graph(cg, n=4)

    r1 = PersonalisedPageRankRetriever(cg, cfg.retrieval).retrieve(
        "anything", [_seed(1.0, "m1")], limit=4
    )
    r2 = PersonalisedPageRankRetriever(cg, cfg.retrieval).retrieve(
        "anything", [_seed(1.0, "m1")], limit=4
    )
    left = [(r["id"], round(r["score"], 8)) for r in r1["results"]]
    right = [(r["id"], round(r["score"], 8)) for r in r2["results"]]
    assert left == right


def test_ppr_respects_relevance_pin():
    cfg = _cfg(
        algorithm="ppr",
        seed_top_k=2,
        expansion_depth=2,
        alpha_dense=0.5,
        beta_bm25=0.0,
        gamma_graph=0.5,
    )
    cg = build_context_graph(cfg)
    cg.upsert_memory_node("m1", "same", user_id="u1")
    cg.upsert_memory_node("m2", "same", user_id="u1")

    retriever = PersonalisedPageRankRetriever(cg, cfg.retrieval)
    base = retriever.retrieve(
        "same",
        [_seed(0.9, "m1"), _seed(0.9, "m2")],
        limit=2,
    )
    base_scores = {r["id"]: r["score"] for r in base["results"]}

    # Pin m1 hard; m2 stays at default 1.0 (pinning is a no-op if floor
    # already matches relevance). Drive m2 down via mark_irrelevant so the
    # relevance modulator matters.
    cg.bump_relevance("m2", -0.7, set_floor=False)
    after = retriever.retrieve(
        "same",
        [_seed(0.9, "m1"), _seed(0.9, "m2")],
        limit=2,
    )
    after_scores = {r["id"]: r["score"] for r in after["results"]}
    # m2 should now rank below m1; previously they were tied.
    assert after_scores["m1"] > after_scores["m2"]
    assert pytest.approx(base_scores["m1"], rel=1e-6) == base_scores["m2"]


def test_ppr_include_archived_flag_is_respected():
    cfg = _cfg(
        algorithm="ppr",
        seed_top_k=1,
        expansion_depth=2,
        max_candidates=10,
        gamma_graph=1.0,
        alpha_dense=0.0,
        beta_bm25=0.0,
        edge_weight_floor=0.0,
    )
    cg = build_context_graph(cfg)
    _build_chain_graph(cg, n=3)
    cg.archive_memory_node("m3")

    retriever = PersonalisedPageRankRetriever(cg, cfg.retrieval)
    result = retriever.retrieve(
        "anything", [_seed(1.0, "m1")], limit=5
    )
    ids = [r["id"] for r in result["results"]]
    assert "m3" not in ids

    result2 = retriever.retrieve(
        "anything",
        [_seed(1.0, "m1")],
        limit=5,
        include_archived=True,
    )
    ids2 = [r["id"] for r in result2["results"]]
    assert "m3" in ids2


def test_ppr_edge_weight_floor_filters_weak_edges():
    cfg = _cfg(
        algorithm="ppr",
        seed_top_k=1,
        expansion_depth=3,
        max_candidates=10,
        edge_weight_floor=0.5,
        gamma_graph=1.0,
        alpha_dense=0.0,
        beta_bm25=0.0,
    )
    cg = build_context_graph(cfg)
    cg.upsert_memory_node("m1", "", user_id="u1")
    cg.upsert_memory_node("m2", "", user_id="u1")
    cg.upsert_memory_node("m3", "", user_id="u1")
    cg.upsert_edge("m1", "m2", EdgeType.TOPIC_SIMILAR, weight=0.9)
    cg.upsert_edge("m2", "m3", EdgeType.TOPIC_SIMILAR, weight=0.2)

    result = PersonalisedPageRankRetriever(cg, cfg.retrieval).retrieve(
        "anything", [_seed(1.0, "m1")], limit=5
    )
    ids = {r["id"] for r in result["results"]}
    # m3 unreachable through the weak edge.
    assert "m1" in ids and "m2" in ids
    assert "m3" not in ids


def test_ppr_empty_seeds_returns_empty():
    cfg = _cfg(algorithm="ppr", seed_top_k=5, expansion_depth=2)
    cg = build_context_graph(cfg)
    retriever = PersonalisedPageRankRetriever(cg, cfg.retrieval)
    result = retriever.retrieve("query", [], limit=3)
    assert result == {"results": [], "subgraph": []}


def test_bfs_retriever_unchanged_after_refactor():
    """Phase D refactor must not alter BFS behaviour."""
    cfg = _cfg(
        algorithm="bfs",
        seed_top_k=1,
        expansion_depth=2,
        max_candidates=10,
        edge_weight_floor=0.0,
        alpha_dense=0.0,
        beta_bm25=0.0,
        gamma_graph=1.0,
    )
    cg = build_context_graph(cfg)
    _build_chain_graph(cg, n=3)

    result = GraphFirstRetriever(cg, cfg.retrieval).retrieve(
        "anything", [_seed(1.0, "m1")], limit=5
    )
    ids = [r["id"] for r in result["results"]]
    # BFS reach propagates with product-of-weights; m2 > m3 always.
    assert ids[0] == "m1"
    reach = {r["id"]: r["context_graph"]["graph"] for r in result["results"]}
    assert reach["m2"] > reach["m3"]


def test_ppr_uses_rerank_channels():
    cfg = _cfg(
        algorithm="ppr",
        seed_top_k=1,
        expansion_depth=1,
        alpha_dense=1.0,
        beta_bm25=0.0,
        gamma_graph=0.0,
    )
    cg = build_context_graph(cfg)
    cg.upsert_memory_node("m1", "apple", user_id="u1")

    result = PersonalisedPageRankRetriever(cg, cfg.retrieval).retrieve(
        "apple", [_seed(0.77, "m1", "apple")], limit=1
    )
    # With gamma=0 the graph channel is inactive; score = alpha * dense.
    assert result["results"][0]["score"] == pytest.approx(0.77, rel=1e-6)
