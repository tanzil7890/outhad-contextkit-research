""" importance scoring from user feedback.

Covers :meth:`Memory.mark_relevant` / :meth:`mark_irrelevant`, the
``bump_relevance`` primitive on the backend ABC, and the sticky
``relevance_floor`` that survives :meth:`ContextGraph.tick_decay`.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.backends.networkx_backend import (
    NetworkXBackend,
)
from outhad_contextkit.memory.context_graph.changelog import ContextChangeLog
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.context_graph.facade import ContextGraph
from outhad_contextkit.memory.context_graph.retriever import GraphFirstRetriever
from outhad_contextkit.memory.context_graph.types import EdgeType


def _cg(**overrides):
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return build_context_graph(cfg)


# ----------------------------------------------------------------------
# facade.bump_relevance
# ----------------------------------------------------------------------


def test_bump_relevance_increases_and_clamps():
    cg = _cg()
    cg.upsert_memory_node("m1", "hello", user_id="u1")
    # Node created with relevance == 1.0 by default. Bump should cap at 1.0.
    r = cg.bump_relevance("m1", 0.3, set_floor=True)
    assert r == pytest.approx(1.0)
    node = cg.backend.get_node("m1")
    assert node.relevance == pytest.approx(1.0)
    # Floor is clamped into [0, 1] too.
    assert 0.0 <= node.relevance_floor <= 1.0


def test_bump_relevance_negative_delta_no_floor():
    cg = _cg()
    cg.upsert_memory_node("m1", "hello", user_id="u1")
    before = cg.backend.get_node("m1").relevance
    r = cg.bump_relevance("m1", -0.4, set_floor=False)
    assert r == pytest.approx(max(0.0, before - 0.4))
    assert cg.backend.get_node("m1").relevance_floor == 0.0


def test_bump_relevance_unknown_node_returns_none():
    cg = _cg()
    assert cg.bump_relevance("ghost", 0.1) is None


def test_bump_relevance_noop_when_disabled():
    cfg = ContextGraphConfig(enabled=False)
    # Disabled path: facade should never be built.
    assert build_context_graph(cfg) is None


# ----------------------------------------------------------------------
# Floor pinned against tick_decay (property-style)
# ----------------------------------------------------------------------


def test_floor_survives_tick_decay():
    backend = NetworkXBackend()
    cfg = ContextGraphConfig(
        enabled=True, backend="networkx", log_changes=False
    )
    cfg.decay.half_life_days = 0.01  # aggressive decay
    cfg.decay.min_node_relevance = 0.05
    cg = ContextGraph(config=cfg, backend=backend, changelog=None)

    cg.upsert_memory_node("pinned", "important", user_id="u1")
    cg.upsert_memory_node("other", "siblings", user_id="u1")
    cg.upsert_edge("pinned", "other", EdgeType.TOPIC_SIMILAR, weight=0.9)

    # Pin the floor well above the decay threshold.
    cg.bump_relevance("pinned", 0.5, set_floor=True)
    floor = cg.backend.get_node("pinned").relevance_floor
    assert floor > cfg.decay.min_node_relevance

    # Age everything so decay has work to do.
    old = datetime.utcnow() - timedelta(days=30)
    for node in list(backend.iter_nodes()):
        node.last_accessed_at = old
        node.updated_at = old
        backend.upsert_node(node)
    for edge in list(backend.all_edges()):
        edge.updated_at = old
        backend.upsert_edge(edge)

    for _ in range(10):
        cg.tick_decay()

    node = backend.get_node("pinned")
    assert node is not None
    assert node.relevance >= floor - 1e-9
    assert node.archived is False


def test_unpinned_node_can_be_archived_by_decay():
    backend = NetworkXBackend()
    cfg = ContextGraphConfig(
        enabled=True, backend="networkx", log_changes=False
    )
    cfg.decay.half_life_days = 0.01
    cfg.decay.min_edge_weight = 0.2
    cfg.decay.min_node_relevance = 0.3
    cg = ContextGraph(config=cfg, backend=backend, changelog=None)

    cg.upsert_memory_node("lonely", "orphan", user_id="u1")
    node = backend.get_node("lonely")
    node.last_accessed_at = datetime.utcnow() - timedelta(days=60)
    node.updated_at = node.last_accessed_at
    backend.upsert_node(node)

    cg.tick_decay()
    assert backend.get_node("lonely").archived is True


# ----------------------------------------------------------------------
# Retriever — pinned seed outranks unpinned one
# ----------------------------------------------------------------------


def test_retriever_boosts_pinned_seed():
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    cg = build_context_graph(cfg)
    cg.upsert_memory_node("pinned", "alpha beta gamma", user_id="u1")
    cg.upsert_memory_node("plain", "alpha beta gamma", user_id="u1")

    # Pin one of them hard.
    cg.bump_relevance("pinned", 0.6, set_floor=True)
    # Knock the other one down so the pinned boost matters.
    plain = cg.backend.get_node("plain")
    plain.relevance = 0.4
    cg.backend.upsert_node(plain)

    retriever = GraphFirstRetriever(cg, cfg.retrieval)
    seeds = [
        {"id": "pinned", "memory": "alpha beta gamma", "score": 0.5},
        {"id": "plain", "memory": "alpha beta gamma", "score": 0.5},
    ]
    out = retriever.retrieve("alpha beta gamma", seeds, limit=2)
    ids = [r["id"] for r in out["results"]]
    assert ids[0] == "pinned"


# ----------------------------------------------------------------------
# Memory facade — sync + async mark_relevant/mark_irrelevant
# ----------------------------------------------------------------------


def test_mark_relevant_returns_none_when_cgl_disabled():
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    obj._context_graph = None
    assert Memory.mark_relevant(obj, "anything") is None
    assert Memory.mark_irrelevant(obj, "anything") is None


def test_mark_relevant_pins_floor_via_memory_facade():
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    obj._context_graph = build_context_graph(cfg)
    obj._context_graph.upsert_memory_node("m1", "hello", user_id="u1")

    # Start relevance below 1.0 so the bump + floor effect is observable.
    node = obj._context_graph.backend.get_node("m1")
    node.relevance = 0.2
    obj._context_graph.backend.upsert_node(node)

    r = Memory.mark_relevant(obj, "m1", delta=0.3)
    assert r == pytest.approx(0.5)
    n = obj._context_graph.backend.get_node("m1")
    assert n.relevance == pytest.approx(0.5)
    assert n.relevance_floor == pytest.approx(0.5)


def test_mark_irrelevant_does_not_pin_floor():
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    obj._context_graph = build_context_graph(cfg)
    obj._context_graph.upsert_memory_node("m1", "hello", user_id="u1")

    r = Memory.mark_irrelevant(obj, "m1", delta=0.4)
    assert r == pytest.approx(0.6)
    n = obj._context_graph.backend.get_node("m1")
    assert n.relevance_floor == 0.0


def test_async_mark_relevant_twin_runs():
    import asyncio

    from outhad_contextkit.memory.main import AsyncMemory

    obj = object.__new__(AsyncMemory)
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    obj._context_graph = build_context_graph(cfg)
    obj._context_graph.upsert_memory_node("m1", "hello", user_id="u1")
    node = obj._context_graph.backend.get_node("m1")
    node.relevance = 0.1
    obj._context_graph.backend.upsert_node(node)

    r = asyncio.run(AsyncMemory.mark_relevant(obj, "m1", delta=0.2))
    assert r == pytest.approx(0.3)
    r2 = asyncio.run(AsyncMemory.mark_irrelevant(obj, "m1", delta=0.1))
    assert r2 == pytest.approx(0.2)


# ----------------------------------------------------------------------
# Changelog emits node_relevance_bumped
# ----------------------------------------------------------------------


def test_changelog_records_bump_event():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = os.path.join(tmpdir, "cgl.db")
        changelog = ContextChangeLog(db)
        backend = NetworkXBackend()
        cfg = ContextGraphConfig(
            enabled=True, backend="networkx", log_changes=True
        )
        cg = ContextGraph(config=cfg, backend=backend, changelog=changelog)
        cg.upsert_memory_node("m1", "hi", user_id="u1")
        cg.bump_relevance("m1", 0.25, set_floor=True)

        events = changelog.recent(target_id="m1", limit=10)
        kinds = [e.event_type for e in events]
        assert "node_relevance_bumped" in kinds
        bump_event = next(e for e in events if e.event_type == "node_relevance_bumped")
        assert bump_event.payload["floor_set"] is True
        assert bump_event.payload["delta"] == pytest.approx(0.25)
