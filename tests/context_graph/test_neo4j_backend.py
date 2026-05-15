"""Neo4j backend integration tests — all marked @pytest.mark.integration.

Skipped automatically when NEO4J_URI is not set in the environment.
Run with a live Neo4j instance:

    NEO4J_URI=bolt://localhost:7687 \
    NEO4J_USERNAME=neo4j \
    NEO4J_PASSWORD=password \
    pytest tests/context_graph/test_neo4j_backend.py -m integration -v

Tests cover the full ABC contract including the methods added for feature
parity: iter_nodes, edges_for_node, set_embedding/get_embedding/iter_embeddings.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from outhad_contextkit.memory.context_graph.backends.neo4j_backend import Neo4jBackend
from outhad_contextkit.memory.context_graph.types import EdgeType, MemoryEdge, MemoryNode

# ---------------------------------------------------------------------------
# Skip guard — no live Neo4j needed for mock-based unit tests below.
# Integration marker gates the live tests.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

_SKIP = pytest.mark.skipif(
    not os.environ.get("NEO4J_URI"),
    reason="NEO4J_URI not set — skipping Neo4j integration tests",
)


# ---------------------------------------------------------------------------
# Mock-based unit tests (no Neo4j required)
# ---------------------------------------------------------------------------

class _MockDriver:
    """Minimal fake that records _execute calls without touching Neo4j."""

    def __init__(self):
        self._calls: list = []
        self._results: list = []  # queue; pop from front per call

    def query(self, q: str, params=None):
        self._calls.append((q, params))
        return self._results.pop(0) if self._results else []

    def push(self, result):
        self._results.append(result)


def _make_backend(driver=None) -> Neo4jBackend:
    if driver is None:
        driver = _MockDriver()
    b = object.__new__(Neo4jBackend)
    b._driver = driver
    return b


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


def _edge(src: str, dst: str, etype: EdgeType = EdgeType.REPLY_TO) -> MemoryEdge:
    return MemoryEdge(src=src, dst=dst, type=etype, weight=1.0)


# ---- iter_nodes ----------------------------------------------------------

def test_iter_nodes_all_returns_nodes():
    driver = _MockDriver()
    driver.push([
        {"props": {
            "id": "m1", "hash": "h", "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(), "archived": False,
            "version": 1, "relevance": 1.0,
        }},
    ])
    b = _make_backend(driver)
    nodes = list(b.iter_nodes(include_archived=True))
    assert len(nodes) == 1
    assert nodes[0].id == "m1"


def test_iter_nodes_exclude_archived_filters_query():
    driver = _MockDriver()
    driver.push([])
    b = _make_backend(driver)
    list(b.iter_nodes(include_archived=False))
    query_used, _ = driver._calls[-1]
    assert "archived" in query_used.lower()


def test_iter_nodes_empty_store_returns_empty():
    driver = _MockDriver()
    driver.push([])
    b = _make_backend(driver)
    assert list(b.iter_nodes()) == []


# ---- edges_for_node ------------------------------------------------------

def test_edges_for_node_returns_incident_edges():
    driver = _MockDriver()
    driver.push([
        {"src": "m1", "dst": "m2", "props": {
            "type": EdgeType.REPLY_TO.value, "weight": 1.0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }},
    ])
    b = _make_backend(driver)
    edges = list(b.edges_for_node("m1"))
    assert len(edges) == 1
    assert edges[0].src == "m1"
    assert edges[0].type == EdgeType.REPLY_TO


def test_edges_for_node_passes_node_id_as_param():
    driver = _MockDriver()
    driver.push([])
    b = _make_backend(driver)
    list(b.edges_for_node("target-id"))
    _, params = driver._calls[-1]
    assert params == {"id": "target-id"}


def test_edges_for_node_empty_returns_empty():
    driver = _MockDriver()
    driver.push([])
    b = _make_backend(driver)
    assert list(b.edges_for_node("nonexistent")) == []


# ---- embedding sidecar ---------------------------------------------------

def test_set_embedding_stores_json():
    driver = _MockDriver()
    b = _make_backend(driver)
    b.set_embedding("m1", [0.1, 0.2, 0.3])
    query, params = driver._calls[-1]
    assert "embedding_json" in query
    import json
    assert json.loads(params["vec"]) == [0.1, 0.2, 0.3]


def test_get_embedding_parses_json():
    import json
    driver = _MockDriver()
    driver.push([{"vec": json.dumps([0.4, 0.5, 0.6])}])
    b = _make_backend(driver)
    vec = b.get_embedding("m1")
    assert vec == pytest.approx([0.4, 0.5, 0.6])


def test_get_embedding_missing_returns_none():
    driver = _MockDriver()
    driver.push([])
    b = _make_backend(driver)
    assert b.get_embedding("missing") is None


def test_get_embedding_null_property_returns_none():
    driver = _MockDriver()
    driver.push([{"vec": None}])
    b = _make_backend(driver)
    assert b.get_embedding("m1") is None


def test_iter_embeddings_returns_id_vec_pairs():
    import json
    driver = _MockDriver()
    driver.push([
        {"id": "m1", "vec": json.dumps([0.1, 0.2])},
        {"id": "m2", "vec": json.dumps([0.3, 0.4])},
    ])
    b = _make_backend(driver)
    pairs = list(b.iter_embeddings())
    assert len(pairs) == 2
    ids = {p[0] for p in pairs}
    assert ids == {"m1", "m2"}


def test_iter_embeddings_skips_null_vec():
    driver = _MockDriver()
    driver.push([{"id": "m1", "vec": None}])
    b = _make_backend(driver)
    assert list(b.iter_embeddings()) == []


def test_iter_embeddings_empty_store():
    driver = _MockDriver()
    driver.push([])
    b = _make_backend(driver)
    assert list(b.iter_embeddings()) == []


# ---------------------------------------------------------------------------
# Live integration tests (require NEO4J_URI)
# ---------------------------------------------------------------------------

@_SKIP
def test_live_iter_nodes_round_trip():
    import neo4j
    driver = neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password")),
    )
    try:
        b = Neo4jBackend(driver)
        node_id = f"test-{uuid.uuid4().hex[:8]}"
        b.upsert_node(_node(node_id))
        nodes = [n for n in b.iter_nodes(include_archived=True) if n.id == node_id]
        assert len(nodes) == 1
        b.delete_node(node_id)
    finally:
        driver.close()


@_SKIP
def test_live_edges_for_node_round_trip():
    import neo4j
    driver = neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password")),
    )
    try:
        b = Neo4jBackend(driver)
        id_a = f"test-{uuid.uuid4().hex[:8]}"
        id_b = f"test-{uuid.uuid4().hex[:8]}"
        b.upsert_node(_node(id_a))
        b.upsert_node(_node(id_b))
        b.upsert_edge(_edge(id_a, id_b, EdgeType.REPLY_TO))
        edges = list(b.edges_for_node(id_a))
        assert any(e.src == id_a and e.dst == id_b for e in edges)
        b.delete_node(id_a)
        b.delete_node(id_b)
    finally:
        driver.close()


@_SKIP
def test_live_embedding_round_trip():
    import neo4j
    driver = neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password")),
    )
    try:
        b = Neo4jBackend(driver)
        node_id = f"test-{uuid.uuid4().hex[:8]}"
        vec = [0.1, 0.2, 0.3, 0.4]
        b.upsert_node(_node(node_id))
        b.set_embedding(node_id, vec)
        fetched = b.get_embedding(node_id)
        assert fetched == pytest.approx(vec)
        pairs = {nid: v for nid, v in b.iter_embeddings() if nid == node_id}
        assert node_id in pairs
        assert pairs[node_id] == pytest.approx(vec)
        b.delete_node(node_id)
    finally:
        driver.close()
