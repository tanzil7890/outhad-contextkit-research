"""Phase 2 — NetworkX backend round-trip contract."""
from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph.backends.networkx_backend import (
    NetworkXBackend,
)
from outhad_contextkit.memory.context_graph.types import EdgeType, MemoryEdge, MemoryNode


def _node(node_id: str, *, archived: bool = False) -> MemoryNode:
    now = datetime.utcnow()
    return MemoryNode(
        id=node_id,
        hash=f"h-{node_id}",
        created_at=now,
        updated_at=now,
        archived=archived,
        user_id="u1",
    )


def _edge(src: str, dst: str, edge_type: EdgeType, weight: float = 1.0) -> MemoryEdge:
    return MemoryEdge(src=src, dst=dst, type=edge_type, weight=weight)


def test_upsert_and_get_node():
    backend = NetworkXBackend()
    node = _node("a")
    backend.upsert_node(node)
    fetched = backend.get_node("a")
    assert fetched is not None
    assert fetched.id == "a"
    assert backend.node_count() == 1


def test_upsert_edge_idempotent():
    backend = NetworkXBackend()
    backend.upsert_node(_node("a"))
    backend.upsert_node(_node("b"))
    backend.upsert_edge(_edge("a", "b", EdgeType.TOPIC_SIMILAR, weight=0.7))
    backend.upsert_edge(_edge("a", "b", EdgeType.TOPIC_SIMILAR, weight=0.9))
    assert backend.edge_count() == 1
    edges = list(backend.all_edges())
    assert edges[0].weight == pytest.approx(0.9)


def test_neighbours_respects_min_weight_and_type():
    backend = NetworkXBackend()
    for nid in ("a", "b", "c"):
        backend.upsert_node(_node(nid))
    backend.upsert_edge(_edge("a", "b", EdgeType.TOPIC_SIMILAR, weight=0.9))
    backend.upsert_edge(_edge("a", "c", EdgeType.REPLY_TO, weight=0.05))

    all_n = backend.neighbours("a", depth=1)
    assert {e.dst for e in all_n} == {"b", "c"}

    filtered = backend.neighbours("a", depth=1, min_weight=0.5)
    assert {e.dst for e in filtered} == {"b"}

    typed = backend.neighbours("a", depth=1, edge_types=[EdgeType.REPLY_TO])
    assert {e.type for e in typed} == {EdgeType.REPLY_TO}


def test_archive_excludes_from_default_traversal():
    backend = NetworkXBackend()
    backend.upsert_node(_node("a"))
    backend.upsert_node(_node("b"))
    backend.upsert_edge(_edge("a", "b", EdgeType.TOPIC_SIMILAR, weight=0.9))
    backend.archive_node("b")

    assert backend.neighbours("a", depth=1) == []
    assert len(backend.neighbours("a", depth=1, include_archived=True)) == 1


def test_remove_edge_and_delete_node():
    backend = NetworkXBackend()
    backend.upsert_node(_node("a"))
    backend.upsert_node(_node("b"))
    backend.upsert_edge(_edge("a", "b", EdgeType.REPLY_TO))
    backend.remove_edge("a", "b", EdgeType.REPLY_TO)
    assert backend.edge_count() == 0

    backend.delete_node("a")
    assert backend.get_node("a") is None
    assert backend.node_count() == 1  # b remains


def test_embeddings_sidecar_round_trip():
    backend = NetworkXBackend()
    backend.upsert_node(_node("a"))
    backend.set_embedding("a", [0.1, 0.2, 0.3])
    assert backend.get_embedding("a") == [0.1, 0.2, 0.3]
    items = list(backend.iter_embeddings())
    assert items == [("a", [0.1, 0.2, 0.3])]


def test_snapshot_and_load_round_trip(tmp_path):
    backend = NetworkXBackend()
    backend.upsert_node(_node("a"))
    backend.upsert_node(_node("b"))
    backend.upsert_edge(_edge("a", "b", EdgeType.DOCUMENT_LINK, weight=0.8))
    backend.set_embedding("a", [1.0, 0.0, 0.0])

    path = tmp_path / "ctx.pkl"
    backend.snapshot(str(path))

    restored = NetworkXBackend()
    restored.load(str(path))
    assert restored.node_count() == 2
    assert restored.edge_count() == 1
    assert restored.get_embedding("a") == [1.0, 0.0, 0.0]


def test_depth_zero_returns_empty():
    backend = NetworkXBackend()
    backend.upsert_node(_node("a"))
    backend.upsert_node(_node("b"))
    backend.upsert_edge(_edge("a", "b", EdgeType.REPLY_TO))
    assert backend.neighbours("a", depth=0) == []
