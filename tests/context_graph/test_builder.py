"""IncrementalGraphBuilder node + edge synthesis contract."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.builder import IncrementalGraphBuilder
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.context_graph.types import EdgeType


@pytest.fixture()
def context_graph():
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    return build_context_graph(cfg)


@pytest.fixture()
def builder(context_graph):
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    return IncrementalGraphBuilder(context_graph, cfg.edges)


def test_on_memory_created_creates_node(context_graph, builder):
    builder.on_memory_created(
        "mem-1",
        "I love pizza",
        user_id="u1",
        run_id="r1",
        metadata={"user_id": "u1", "run_id": "r1"},
    )
    node = context_graph.backend.get_node("mem-1")
    assert node is not None
    assert node.user_id == "u1"
    assert node.run_id == "r1"


def test_reply_to_edge_within_window(context_graph, builder):
    t0 = datetime.utcnow()
    builder.on_memory_created(
        "mem-1",
        "first",
        user_id="u1",
        run_id="r1",
        created_at=t0,
    )
    builder.on_memory_created(
        "mem-2",
        "second",
        user_id="u1",
        run_id="r1",
        created_at=t0 + timedelta(seconds=5),
    )
    neighbours = context_graph.backend.neighbours("mem-1", depth=1)
    edge_types = {e.type for e in neighbours}
    assert EdgeType.REPLY_TO in edge_types


def test_reply_to_suppressed_outside_window(context_graph):
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    cfg.edges.reply_to_window_seconds = 10
    builder = IncrementalGraphBuilder(context_graph, cfg.edges)
    t0 = datetime.utcnow()
    builder.on_memory_created("m1", "a", run_id="r1", user_id="u1", created_at=t0)
    builder.on_memory_created(
        "m2",
        "b",
        run_id="r1",
        user_id="u1",
        created_at=t0 + timedelta(seconds=60),
    )
    types = {e.type for e in context_graph.backend.neighbours("m1", depth=1)}
    assert EdgeType.REPLY_TO not in types
    # Temporal next edge still fires regardless of window.
    assert EdgeType.TEMPORAL_NEXT in types


def test_document_link_by_metadata(context_graph, builder):
    builder.on_memory_created(
        "m1",
        "alpha",
        user_id="u1",
        metadata={"document_id": "doc-42"},
    )
    builder.on_memory_created(
        "m2",
        "beta",
        user_id="u1",
        metadata={"document_id": "doc-42"},
    )
    types = {e.type for e in context_graph.backend.neighbours("m2", depth=1)}
    assert EdgeType.DOCUMENT_LINK in types


def test_topic_similar_edge(context_graph, builder):
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.95, 0.05, 0.0]
    emb_c = [0.0, 1.0, 0.0]
    builder.on_memory_created("m1", "cats", user_id="u1", embedding=emb_a)
    builder.on_memory_created("m2", "kittens", user_id="u1", embedding=emb_b)
    builder.on_memory_created("m3", "trains", user_id="u1", embedding=emb_c)

    neighbours = context_graph.backend.neighbours("m2", depth=1)
    topic_edges = [e for e in neighbours if e.type == EdgeType.TOPIC_SIMILAR]
    assert topic_edges, "expected at least one TOPIC_SIMILAR edge"


def test_on_memory_deleted_soft_archives(context_graph, builder):
    builder.on_memory_created("m1", "hello", user_id="u1")
    builder.on_memory_deleted("m1", user_id="u1", hard=False)
    node = context_graph.backend.get_node("m1")
    assert node is not None
    assert node.archived is True


def test_on_memory_deleted_hard_removes(context_graph, builder):
    builder.on_memory_created("m1", "hello", user_id="u1")
    builder.on_memory_deleted("m1", user_id="u1", hard=True)
    assert context_graph.backend.get_node("m1") is None


def test_on_memory_updated_bumps_version(context_graph, builder):
    builder.on_memory_created("m1", "v1 content", user_id="u1")
    first = context_graph.backend.get_node("m1")
    assert first.version == 1
    builder.on_memory_updated("m1", "v2 content", user_id="u1")
    second = context_graph.backend.get_node("m1")
    assert second.version == 2
