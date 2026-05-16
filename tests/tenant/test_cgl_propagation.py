""" CGL tenant propagation tests."""
from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph.backends import (
    _dict_to_node,
    _node_to_dict,
)
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


def _now():
    return datetime(2026, 4, 25, 12, 0, 0)


def _make_graph():
    backend = NetworkXBackend()
    cg = ContextGraph(
        config=ContextGraphConfig(enabled=True, log_changes=False),
        backend=backend,
        changelog=None,
    )
    return backend, cg


# ---------------------------------------------------------------------------
# MemoryNode field defaults
# ---------------------------------------------------------------------------

def test_memory_node_tenant_defaults_none():
    n = MemoryNode(id="m1", hash="h", created_at=_now(), updated_at=_now())
    assert n.tenant_id is None
    assert n.sub_tenant_id is None


def test_memory_node_tenant_round_trip_dict():
    n = MemoryNode(
        id="m1",
        hash="h",
        created_at=_now(),
        updated_at=_now(),
        tenant_id="acme",
        sub_tenant_id="finance",
    )
    data = _node_to_dict(n)
    assert data["tenant_id"] == "acme"
    assert data["sub_tenant_id"] == "finance"
    restored = _dict_to_node(data)
    assert restored.tenant_id == "acme"
    assert restored.sub_tenant_id == "finance"


def test_legacy_node_dict_loads_with_null_tenant():
    """Pre-T5 serialised nodes lack tenant keys → load as None."""
    legacy = {
        "id": "m1",
        "hash": "h",
        "created_at": _now().isoformat(),
        "updated_at": _now().isoformat(),
    }
    node = _dict_to_node(legacy)
    assert node.tenant_id is None
    assert node.sub_tenant_id is None


# ---------------------------------------------------------------------------
# Facade upsert propagates tenant
# ---------------------------------------------------------------------------

def test_facade_upsert_propagates_explicit_tenant():
    _, cg = _make_graph()
    cg.upsert_memory_node(
        "m1", "hello", tenant_id="acme", sub_tenant_id="finance"
    )
    node = cg.backend.get_node("m1")
    assert node.tenant_id == "acme"
    assert node.sub_tenant_id == "finance"


def test_facade_upsert_falls_back_to_metadata_tenant():
    """When the caller stuffs tenant in metadata, facade extracts it."""
    _, cg = _make_graph()
    cg.upsert_memory_node(
        "m1",
        "hello",
        metadata={"tenant_id": "acme", "sub_tenant_id": "finance"},
    )
    node = cg.backend.get_node("m1")
    assert node.tenant_id == "acme"
    assert node.sub_tenant_id == "finance"


def test_facade_explicit_tenant_overrides_metadata():
    _, cg = _make_graph()
    cg.upsert_memory_node(
        "m1",
        "hello",
        tenant_id="acme_override",
        metadata={"tenant_id": "metadata_value"},
    )
    assert cg.backend.get_node("m1").tenant_id == "acme_override"


def test_facade_update_does_not_clear_tenant_with_null():
    """Subsequent upsert with tenant_id=None must not erase a tagged node."""
    _, cg = _make_graph()
    cg.upsert_memory_node("m1", "hello", tenant_id="acme")
    cg.upsert_memory_node("m1", "hello v2")  # no tenant in second call
    assert cg.backend.get_node("m1").tenant_id == "acme"


# ---------------------------------------------------------------------------
# Builder propagates tenant
# ---------------------------------------------------------------------------

def test_builder_passes_tenant_to_facade():
    pytest.importorskip("networkx")
    from outhad_contextkit.memory.context_graph.builder import (
        IncrementalGraphBuilder,
    )
    from outhad_contextkit.memory.context_graph.config import (
        EdgeSynthesisConfig,
    )

    _, cg = _make_graph()
    builder = IncrementalGraphBuilder(cg, EdgeSynthesisConfig())
    builder.on_memory_created(
        "m1",
        "hello",
        user_id="alice",
        tenant_id="acme",
        sub_tenant_id="finance",
    )
    node = cg.backend.get_node("m1")
    assert node.tenant_id == "acme"
    assert node.sub_tenant_id == "finance"


# ---------------------------------------------------------------------------
# Retriever hard-skip filter
# ---------------------------------------------------------------------------

def test_retriever_hard_skips_cross_tenant_nodes():
    _, cg = _make_graph()
    cg.upsert_memory_node("a", "alpha", tenant_id="acme")
    cg.upsert_memory_node("b", "beta", tenant_id="other")
    seeds = [
        {"id": "a", "memory": "alpha", "score": 0.9},
        {"id": "b", "memory": "beta", "score": 0.8},
    ]
    cfg = RetrievalConfig(
        alpha_dense=1.0, beta_bm25=0.0, gamma_graph=0.0, expansion_depth=0
    )
    out = GraphFirstRetriever(cg, cfg).retrieve(
        "q", seeds, limit=5, tenant_id="acme"
    )
    ids = [r["id"] for r in out["results"]]
    assert ids == ["a"]


def test_retriever_null_tenant_nodes_pass_filter():
    """Legacy nodes (tenant_id IS NULL) survive scoped retrieval."""
    _, cg = _make_graph()
    cg.upsert_memory_node("a", "alpha", tenant_id="acme")
    cg.upsert_memory_node("b", "beta")  # legacy / NULL
    seeds = [
        {"id": "a", "memory": "alpha", "score": 0.5},
        {"id": "b", "memory": "beta", "score": 0.5},
    ]
    cfg = RetrievalConfig(
        alpha_dense=1.0, beta_bm25=0.0, gamma_graph=0.0, expansion_depth=0
    )
    out = GraphFirstRetriever(cg, cfg).retrieve(
        "q", seeds, limit=5, tenant_id="acme"
    )
    ids = sorted(r["id"] for r in out["results"])
    assert ids == ["a", "b"]


def test_retriever_no_tenant_filter_returns_all():
    """tenant_id=None → no filter; pre-T5 behaviour preserved."""
    _, cg = _make_graph()
    cg.upsert_memory_node("a", "alpha", tenant_id="acme")
    cg.upsert_memory_node("b", "beta", tenant_id="other")
    seeds = [
        {"id": "a", "memory": "alpha", "score": 0.5},
        {"id": "b", "memory": "beta", "score": 0.5},
    ]
    cfg = RetrievalConfig(
        alpha_dense=1.0, beta_bm25=0.0, gamma_graph=0.0, expansion_depth=0
    )
    out = GraphFirstRetriever(cg, cfg).retrieve("q", seeds, limit=5)
    assert len(out["results"]) == 2


def test_retriever_sub_tenant_filter():
    _, cg = _make_graph()
    cg.upsert_memory_node(
        "a", "alpha", tenant_id="acme", sub_tenant_id="finance"
    )
    cg.upsert_memory_node(
        "b", "beta", tenant_id="acme", sub_tenant_id="engineering"
    )
    seeds = [
        {"id": "a", "memory": "alpha", "score": 0.5},
        {"id": "b", "memory": "beta", "score": 0.5},
    ]
    cfg = RetrievalConfig(
        alpha_dense=1.0, beta_bm25=0.0, gamma_graph=0.0, expansion_depth=0
    )
    out = GraphFirstRetriever(cg, cfg).retrieve(
        "q", seeds, limit=5, tenant_id="acme", sub_tenant_id="finance"
    )
    ids = [r["id"] for r in out["results"]]
    assert ids == ["a"]
