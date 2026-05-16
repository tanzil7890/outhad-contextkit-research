""" ``Memory._cgl_on_*`` hook semantics (without booting Memory stack).

These tests rebind the hook methods onto a bare object and exercise them with
a real builder + backend. That confirms the hooks drive the CGL exactly as the
production call sites expect, while staying independent of the vector store,
LLM, and embedding model that a full ``Memory()`` would require.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.builder import IncrementalGraphBuilder
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.main import Memory


def _bare_memory():
    obj = object.__new__(Memory)
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    cg = build_context_graph(cfg)
    obj._context_graph = cg
    obj._context_graph_builder = IncrementalGraphBuilder(cg, cfg.edges)
    return obj, cg


def test_cgl_on_created_creates_node():
    obj, cg = _bare_memory()
    Memory._cgl_on_created(
        obj,
        "mem-1",
        "hello world",
        [0.1, 0.2, 0.3],
        {"user_id": "u1", "run_id": "r1"},
    )
    assert cg.backend.get_node("mem-1") is not None


def test_cgl_on_updated_bumps_version():
    obj, cg = _bare_memory()
    Memory._cgl_on_created(obj, "mem-1", "v1", None, {"user_id": "u1"})
    Memory._cgl_on_updated(obj, "mem-1", "v2", None, {"user_id": "u1"}, prev_value="v1")
    node = cg.backend.get_node("mem-1")
    assert node.version == 2


def test_cgl_on_deleted_soft_archives():
    obj, cg = _bare_memory()
    Memory._cgl_on_created(obj, "mem-1", "x", None, {"user_id": "u1"})
    Memory._cgl_on_deleted(obj, "mem-1", {"user_id": "u1"})
    node = cg.backend.get_node("mem-1")
    assert node is not None and node.archived is True


def test_hooks_no_op_when_builder_absent():
    obj = object.__new__(Memory)
    obj._context_graph = None
    obj._context_graph_builder = None
    # Must not raise.
    Memory._cgl_on_created(obj, "id", "data", None, None)
    Memory._cgl_on_updated(obj, "id", "data", None, None)
    Memory._cgl_on_deleted(obj, "id", None)


def test_hook_exceptions_are_swallowed(monkeypatch):
    obj, cg = _bare_memory()

    def boom(*_, **__):
        raise RuntimeError("backend is on fire")

    # Force the builder to explode and assert the hook stays quiet.
    obj._context_graph_builder.on_memory_created = types.MethodType(
        boom, obj._context_graph_builder
    )
    Memory._cgl_on_created(obj, "m", "d", None, None)  # must not raise
