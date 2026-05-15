"""Phase 3 — ``ContextGraph`` facade: decay, stats, persistence."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.backends.networkx_backend import (
    NetworkXBackend,
)
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.context_graph.facade import ContextGraph
from outhad_contextkit.memory.context_graph.types import EdgeType


def _cg(**overrides):
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return build_context_graph(cfg)


def test_upsert_memory_node_idempotent():
    cg = _cg()
    node_a = cg.upsert_memory_node("m1", "hi", user_id="u1")
    node_b = cg.upsert_memory_node("m1", "hi", user_id="u1")
    assert node_a.id == node_b.id
    assert node_b.version == 1


def test_version_bumps_when_hash_changes():
    cg = _cg()
    cg.upsert_memory_node("m1", "first", user_id="u1")
    node = cg.upsert_memory_node("m1", "second", user_id="u1")
    assert node.version == 2
    assert node.prev_version_id == "m1"


def test_tick_decay_prunes_low_weight_edges():
    backend = NetworkXBackend()
    cfg = ContextGraphConfig(
        enabled=True, backend="networkx", log_changes=False
    )
    cfg.decay.half_life_days = 1.0
    cfg.decay.min_edge_weight = 0.2
    cg = ContextGraph(config=cfg, backend=backend, changelog=None)

    cg.upsert_memory_node("m1", "a", user_id="u1")
    cg.upsert_memory_node("m2", "b", user_id="u1")
    edge = cg.upsert_edge("m1", "m2", EdgeType.TOPIC_SIMILAR, weight=1.0)

    # Force the edge to look old (updated_at 10 days ago → ~2^-10 < 0.2).
    edge.updated_at = datetime.utcnow() - timedelta(days=10)
    backend.upsert_edge(edge)

    stats = cg.tick_decay()
    assert stats["pruned"] >= 1
    assert backend.edge_count() == 0


def test_causal_edges_are_sticky():
    backend = NetworkXBackend()
    cfg = ContextGraphConfig(
        enabled=True, backend="networkx", log_changes=False
    )
    cfg.decay.half_life_days = 0.01
    cg = ContextGraph(config=cfg, backend=backend, changelog=None)

    cg.upsert_memory_node("m1", "a", user_id="u1")
    cg.upsert_memory_node("m2", "b", user_id="u1")
    cg.upsert_edge("m1", "m2", EdgeType.CAUSAL, weight=0.9)

    cg.tick_decay()
    assert backend.edge_count() == 1


def test_snapshot_round_trip(tmp_path):
    path = tmp_path / "snap.pkl"
    cfg = ContextGraphConfig(
        enabled=True,
        backend="networkx",
        log_changes=False,
        persist_path=str(path),
    )
    cg = build_context_graph(cfg)
    cg.upsert_memory_node("m1", "alpha", user_id="u1")
    cg.snapshot()
    assert path.exists()

    # Re-open a new facade against the same snapshot; nodes must persist.
    cg2 = build_context_graph(cfg)
    assert cg2.stats()["nodes"] == 1
