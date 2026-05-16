""" MemoryNode round-trip covers the new MSPR fields.

Both backends must preserve ``access_count``, ``helpful_count``,
``unhelpful_count``, and ``last_feedback_at`` through serialise → persist →
deserialise cycles. Legacy payloads missing those keys must default to
zero / None without raising.
"""
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
from outhad_contextkit.memory.context_graph.types import MemoryNode


def _now() -> datetime:
    return datetime(2026, 4, 24, 10, 0, 0)


def test_node_to_dict_includes_mspr_fields():
    node = MemoryNode(
        id="m1",
        hash="h",
        created_at=_now(),
        updated_at=_now(),
        access_count=7,
        helpful_count=3,
        unhelpful_count=1,
        last_feedback_at=_now(),
    )
    data = _node_to_dict(node)
    assert data["access_count"] == 7
    assert data["helpful_count"] == 3
    assert data["unhelpful_count"] == 1
    assert data["last_feedback_at"] == _now().isoformat()


def test_dict_to_node_defaults_zero_when_missing():
    node = _dict_to_node({"id": "legacy", "hash": "h"})
    assert node.access_count == 0
    assert node.helpful_count == 0
    assert node.unhelpful_count == 0
    assert node.last_feedback_at is None


def test_dict_to_node_preserves_new_fields():
    data = {
        "id": "m2",
        "hash": "h",
        "access_count": 5,
        "helpful_count": 2,
        "unhelpful_count": 4,
        "last_feedback_at": _now().isoformat(),
    }
    node = _dict_to_node(data)
    assert node.access_count == 5
    assert node.helpful_count == 2
    assert node.unhelpful_count == 4
    assert node.last_feedback_at == _now()


def test_networkx_backend_preserves_counters_through_snapshot(tmp_path):
    backend = NetworkXBackend()
    node = MemoryNode(
        id="m3",
        hash="h",
        created_at=_now(),
        updated_at=_now(),
        access_count=9,
        helpful_count=4,
    )
    backend.upsert_node(node)

    path = str(tmp_path / "export.jsonl")
    backend.export_jsonl(path)

    other = NetworkXBackend()
    stats = other.import_jsonl(path)
    assert stats["nodes"] == 1
    restored = other.get_node("m3")
    assert restored is not None
    assert restored.access_count == 9
    assert restored.helpful_count == 4
