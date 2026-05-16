"""backfill_context_graph tests.

Verifies:
1. Nodes get created for every memory in the mock vector store.
2. The call is idempotent (second backfill does not duplicate nodes).
3. RuntimeError is raised when CGL is disabled.
4. batch_size slicing is exercised (large memory list, small batch).
5. embed=False skips embedding calls.
6. Scoped backfill (user_id filter plumbed through to vector_store.list).
"""
from __future__ import annotations

import types
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.builder import IncrementalGraphBuilder
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.main import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeMemItem:
    def __init__(self, id: str, data: str, user_id: str = "u1", run_id: str = "r1"):
        self.id = id
        self.payload: Dict[str, Any] = {
            "data": data,
            "user_id": user_id,
            "run_id": run_id,
            "created_at": datetime.utcnow().isoformat(),
        }


def _bare_memory_with_store(memories: list):
    obj = object.__new__(Memory)
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    cg = build_context_graph(cfg)
    obj._context_graph = cg
    builder = IncrementalGraphBuilder(cg, cfg.edges)
    obj._context_graph_builder = builder
    obj.config = MagicMock()
    obj.config.context_graph = cfg

    mock_store = MagicMock()
    mock_store.list.return_value = [memories]
    obj.vector_store = mock_store

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    obj.embedding_model = mock_embedder

    return obj, cg, mock_store, mock_embedder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_backfill_creates_nodes_for_all_memories():
    mems = [_FakeMemItem(f"m{i}", f"text {i}") for i in range(5)]
    obj, cg, _, _ = _bare_memory_with_store(mems)
    result = Memory.backfill_context_graph(obj)
    assert result["nodes"] == 5
    for mem in mems:
        assert cg.backend.get_node(mem.id) is not None


def test_backfill_is_idempotent():
    mems = [_FakeMemItem("m1", "hello"), _FakeMemItem("m2", "world")]
    obj, cg, _, _ = _bare_memory_with_store(mems)
    r1 = Memory.backfill_context_graph(obj)
    r2 = Memory.backfill_context_graph(obj)
    assert r1["nodes"] == r2["nodes"] == 2
    assert cg.backend.node_count(include_archived=False) == 2


def test_backfill_raises_when_cgl_disabled():
    obj = object.__new__(Memory)
    obj._context_graph = None
    obj._context_graph_builder = None
    with pytest.raises(RuntimeError, match="not enabled"):
        Memory.backfill_context_graph(obj)


def test_backfill_respects_batch_size():
    mems = [_FakeMemItem(f"m{i}", f"text {i}") for i in range(10)]
    obj, cg, _, embedder = _bare_memory_with_store(mems)
    result = Memory.backfill_context_graph(obj, batch_size=3)
    assert result["nodes"] == 10
    # All 10 embeddings should have been computed (embed=True default)
    assert embedder.embed.call_count == 10


def test_backfill_embed_false_skips_embedding():
    mems = [_FakeMemItem("m1", "text")]
    obj, cg, _, embedder = _bare_memory_with_store(mems)
    Memory.backfill_context_graph(obj, embed=False)
    embedder.embed.assert_not_called()


def test_backfill_scoped_filter_passed_to_vector_store():
    mems = [_FakeMemItem("m1", "text", user_id="alice")]
    obj, cg, store, _ = _bare_memory_with_store(mems)
    Memory.backfill_context_graph(obj, user_id="alice")
    store.list.assert_called_once_with(filters={"user_id": "alice"}, limit=100_000)


def test_backfill_no_filter_calls_store_with_none():
    mems = [_FakeMemItem("m1", "text")]
    obj, cg, store, _ = _bare_memory_with_store(mems)
    Memory.backfill_context_graph(obj)
    store.list.assert_called_once_with(filters=None, limit=100_000)


def test_backfill_empty_store_returns_zero():
    obj, cg, store, _ = _bare_memory_with_store([])
    store.list.return_value = [[]]
    result = Memory.backfill_context_graph(obj)
    assert result == {"nodes": 0, "edges": 0}


def test_backfill_edges_reported():
    # Two memories in same run → REPLY_TO + TEMPORAL_NEXT edges should form.
    mems = [
        _FakeMemItem("m1", "first", user_id="u1", run_id="r1"),
        _FakeMemItem("m2", "second", user_id="u1", run_id="r1"),
    ]
    obj, cg, _, _ = _bare_memory_with_store(mems)
    result = Memory.backfill_context_graph(obj)
    assert result["nodes"] == 2
    assert result["edges"] >= 1  # at least REPLY_TO or TEMPORAL_NEXT
