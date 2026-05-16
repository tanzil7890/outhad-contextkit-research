""" property tests (Hypothesis) for the Context-Graph Layer.

Two invariants the layer must preserve under arbitrary operation orderings:

1. **Backend node count**: ``live_nodes == node_added − node_archived −
   hard_deleted`` for any sequence of upsert/archive/delete operations.
2. **Decay monotonicity**: ``tick_decay()`` never increases an edge weight
   nor un-archives a node.

Hypothesis is an optional dev dep; the tests skip cleanly if it is missing.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

pytest.importorskip("networkx")
hyp = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from outhad_contextkit.memory.context_graph.backends.networkx_backend import (  # noqa: E402
    NetworkXBackend,
)
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig  # noqa: E402
from outhad_contextkit.memory.context_graph.facade import ContextGraph  # noqa: E402
from outhad_contextkit.memory.context_graph.types import EdgeType  # noqa: E402


def _new_graph() -> ContextGraph:
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    cfg.decay.half_life_days = 1.0
    cfg.decay.min_edge_weight = 0.05
    cfg.decay.min_node_relevance = 0.05
    return ContextGraph(config=cfg, backend=NetworkXBackend(), changelog=None)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(
    ops=st.lists(
        st.tuples(
            st.sampled_from(["add", "archive", "delete"]),
            st.integers(min_value=0, max_value=6),  # 7 candidate node ids
        ),
        min_size=1,
        max_size=30,
    )
)
def test_node_accounting_invariant(ops):
    """live = added − archived − deleted, regardless of operation order."""
    cg = _new_graph()
    backend = cg.backend

    # Model node lifecycle exactly like the backend does:
    #   - "add" creates a node (idempotent on id; does NOT un-archive)
    #   - "archive" sets archived=True (no-op if already archived/deleted)
    #   - "delete" removes the node entirely
    present: set[str] = set()         # node exists in backend (any state)
    archived: set[str] = set()        # subset of present that is archived

    for op, idx in ops:
        node_id = f"n{idx}"
        if op == "add":
            cg.upsert_memory_node(node_id, f"text-{idx}", user_id="u")
            present.add(node_id)
            # second add on an archived node leaves it archived (matches facade).
        elif op == "archive":
            if node_id in present:
                cg.archive_memory_node(node_id)
                archived.add(node_id)
        elif op == "delete":
            if node_id in present:
                cg.delete_memory_node(node_id)
                present.discard(node_id)
                archived.discard(node_id)

    live_expected = len(present - archived)
    live_actual = backend.node_count(include_archived=False)
    assert live_actual == live_expected, (
        f"live mismatch: backend={live_actual}, expected={live_expected}, "
        f"present={present}, archived={archived}"
    )


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    edges=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=4),     # src
            st.integers(min_value=0, max_value=4),     # dst
            st.floats(min_value=0.1, max_value=1.0),   # initial weight
            st.integers(min_value=1, max_value=240),   # hours of age
        ),
        min_size=1,
        max_size=12,
    )
)
def test_tick_decay_is_monotonic(edges):
    """Weight after tick_decay() is never greater than weight before."""
    cg = _new_graph()
    backend = cg.backend

    for src, dst, weight, _ in edges:
        if src == dst:
            continue  # graph is a DAG-of-pairs; skip self-loops
        cg.upsert_memory_node(f"n{src}", f"text-{src}", user_id="u")
        cg.upsert_memory_node(f"n{dst}", f"text-{dst}", user_id="u")
        cg.upsert_edge(
            f"n{src}", f"n{dst}", EdgeType.TOPIC_SIMILAR, weight=float(weight)
        )

    # Back-date every edge so decay has work to do.
    now = datetime.utcnow()
    pre_weights: dict[tuple[str, str, str], float] = {}
    for edge, (_, _, _, hours) in zip(list(backend.all_edges()), edges):
        edge.updated_at = now - timedelta(hours=int(hours))
        backend.upsert_edge(edge)
        pre_weights[(edge.src, edge.dst, edge.type.value)] = edge.weight

    pre_archived = {
        n.id for n in backend.iter_nodes(include_archived=True) if n.archived
    }

    cg.tick_decay(now)

    post_archived = {
        n.id for n in backend.iter_nodes(include_archived=True) if n.archived
    }
    # Archived set only grows under decay.
    assert pre_archived.issubset(post_archived)

    for edge in backend.all_edges():
        key = (edge.src, edge.dst, edge.type.value)
        prior = pre_weights.get(key)
        if prior is None:
            continue  # edge was pruned and re-created via re-synthesis (n/a here)
        assert edge.weight <= prior + 1e-9, (
            f"edge {key} weight grew: {prior} -> {edge.weight}"
        )
