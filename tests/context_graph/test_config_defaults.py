"""Phase 1 — Configuration defaults and opt-in semantics.

These tests guarantee that enabling the Context-Graph Layer is always an
explicit choice. The feature must remain dormant (byte-identical behaviour)
until the user flips ``context_graph.enabled``.
"""
from __future__ import annotations

from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.context_graph import (
    ContextGraphConfig,
    EdgeType,
    build_context_graph,
)
from outhad_contextkit.memory.context_graph.config import (
    DecayConfig,
    EdgeSynthesisConfig,
    RetrievalConfig,
)


def test_context_graph_disabled_by_default():
    cfg = MemoryConfig()
    assert cfg.context_graph.enabled is False
    assert cfg.context_graph.backend == "networkx"


def test_sub_config_defaults_are_safe():
    cfg = ContextGraphConfig()
    assert isinstance(cfg.decay, DecayConfig)
    assert isinstance(cfg.edges, EdgeSynthesisConfig)
    assert isinstance(cfg.retrieval, RetrievalConfig)

    assert cfg.decay.enabled is True
    assert cfg.decay.half_life_days > 0
    assert 0 <= cfg.decay.min_edge_weight <= 1

    assert cfg.edges.enable_reply_to is True
    assert cfg.edges.enable_topic_similar is True
    assert cfg.edges.enable_document_link is True
    assert cfg.edges.enable_temporal_next is True
    assert cfg.edges.topic_top_k >= 1
    assert 0 <= cfg.edges.topic_min_similarity <= 1

    assert cfg.retrieval.seed_top_k >= 1
    assert 0 <= cfg.retrieval.expansion_depth <= 4


def test_edge_type_enum_values_stable():
    # The values are part of the public surface (cypher / pickle); freeze them.
    assert EdgeType.REPLY_TO.value == "REPLY_TO"
    assert EdgeType.TOPIC_SIMILAR.value == "TOPIC_SIMILAR"
    assert EdgeType.DOCUMENT_LINK.value == "DOCUMENT_LINK"
    assert EdgeType.TEMPORAL_NEXT.value == "TEMPORAL_NEXT"
    assert EdgeType.UPDATED_FROM.value == "UPDATED_FROM"
    assert EdgeType.CAUSAL.value == "CAUSAL"


def test_build_context_graph_returns_none_when_disabled():
    assert build_context_graph(ContextGraphConfig(enabled=False)) is None


def test_build_context_graph_networkx_smoke(tmp_path):
    cfg = ContextGraphConfig(
        enabled=True,
        backend="networkx",
        persist_path=str(tmp_path / "snapshot.pkl"),
        log_changes=False,
    )
    graph = build_context_graph(cfg)
    assert graph is not None
    stats = graph.stats()
    assert stats == {"nodes": 0, "nodes_total": 0, "edges": 0}
