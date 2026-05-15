"""Phase 3/5 — SQLite change timeline writer/reader contract."""
from __future__ import annotations

from datetime import datetime

from outhad_contextkit.memory.context_graph.changelog import ContextChangeLog
from outhad_contextkit.memory.context_graph.types import ChangeEvent


def test_append_and_read_recent(tmp_path):
    log = ContextChangeLog(str(tmp_path / "changes.db"))
    log.append(
        ChangeEvent(
            event_type="node_added",
            target_id="mem-1",
            timestamp=datetime.utcnow(),
            user_id="u1",
            payload={"hash": "abc"},
        )
    )
    log.append(
        ChangeEvent(
            event_type="edge_added",
            target_id="mem-1->mem-2",
            timestamp=datetime.utcnow(),
            user_id="u1",
            payload={"type": "REPLY_TO"},
        )
    )
    events = log.recent(limit=10)
    assert len(events) == 2
    assert {e.event_type for e in events} == {"node_added", "edge_added"}


def test_recent_filters_by_target(tmp_path):
    log = ContextChangeLog(str(tmp_path / "c.db"))
    log.append(
        ChangeEvent(
            event_type="node_added",
            target_id="A",
            timestamp=datetime.utcnow(),
            payload={},
        )
    )
    log.append(
        ChangeEvent(
            event_type="node_added",
            target_id="B",
            timestamp=datetime.utcnow(),
            payload={},
        )
    )
    a_events = log.recent(target_id="A")
    assert len(a_events) == 1
    assert a_events[0].target_id == "A"


def test_schema_is_idempotent(tmp_path):
    db_path = tmp_path / "c.db"
    ContextChangeLog(str(db_path))
    ContextChangeLog(str(db_path))  # second bootstrap must not raise
