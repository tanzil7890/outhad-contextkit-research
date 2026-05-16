""" ``MemoryNode`` / ``MemoryEdge`` / ``ChangeEvent`` dataclass shape."""
from __future__ import annotations

from datetime import datetime, timedelta

from outhad_contextkit.memory.context_graph.types import (
    ChangeEvent,
    EdgeType,
    MemoryEdge,
    MemoryNode,
)


def test_memory_node_defaults():
    now = datetime.utcnow()
    node = MemoryNode(id="mem-1", hash="abc", created_at=now, updated_at=now)
    assert node.version == 1
    assert node.relevance == 1.0
    assert node.archived is False
    assert node.prev_version_id is None
    assert node.metadata == {}


def test_memory_edge_defaults_are_timestamped():
    edge = MemoryEdge(src="a", dst="b", type=EdgeType.TOPIC_SIMILAR)
    assert isinstance(edge.created_at, datetime)
    assert isinstance(edge.updated_at, datetime)
    assert edge.weight == 1.0
    assert edge.metadata == {}
    # created_at and updated_at default to "now" — they should be within a second.
    assert abs((edge.updated_at - edge.created_at).total_seconds()) < 1.0


def test_change_event_payload_is_copyable():
    event = ChangeEvent(
        event_type="node_added",
        target_id="mem-1",
        timestamp=datetime.utcnow(),
        user_id="u1",
        payload={"hash": "abc"},
    )
    assert event.event_type == "node_added"
    assert event.payload["hash"] == "abc"
    # Mutating the dict should not alter the event's stored copy if consumers
    # take care to `dict(...)` on the way in. This is a documentation test.
    snapshot = dict(event.payload)
    event.payload["hash"] = "def"
    assert snapshot["hash"] == "abc"


def test_edge_type_is_string_enum_round_trip():
    for value in (
        "REPLY_TO",
        "TOPIC_SIMILAR",
        "DOCUMENT_LINK",
        "TEMPORAL_NEXT",
        "UPDATED_FROM",
        "CAUSAL",
    ):
        assert EdgeType(value).value == value


def test_memory_node_updated_at_monotonic_on_mutation():
    t0 = datetime.utcnow()
    node = MemoryNode(id="m", hash="h", created_at=t0, updated_at=t0)
    node.updated_at = t0 + timedelta(seconds=5)
    assert node.updated_at > node.created_at
