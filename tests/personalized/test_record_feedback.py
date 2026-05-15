"""Phase F3 — ``Memory.record_feedback`` integration (no vector store).

Builds a minimal Memory shell (no LLM / vector store / embedder) by
constructing only the CGL + MSPR attributes the method touches, then
asserting the full contract: relevance bump + SQLite row + change-bus
event + counter updates + clamp behaviour.
"""
from __future__ import annotations

from datetime import datetime
from typing import List

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.context_graph.types import MemoryNode
from outhad_contextkit.memory.main import Memory
from outhad_contextkit.memory.personalized.feedback_store import FeedbackStore


class _EventSpy:
    def __init__(self):
        self.events: List[dict] = []

    def __call__(self, ev):
        self.events.append(
            {"type": ev.event_type, "target": ev.target_id, "payload": ev.payload}
        )


def _now():
    return datetime(2026, 4, 24, 12, 0, 0)


def _bare_memory(tmp_path):
    obj = object.__new__(Memory)
    cfg = MemoryConfig()
    cfg.context_graph = ContextGraphConfig(
        enabled=True, backend="networkx", log_changes=False
    )
    cfg.mspr.enabled = True
    cfg.mspr.feedback.enabled = True
    cfg.mspr.feedback.boost_delta = 0.25
    cfg.mspr.feedback.penalty_delta = 0.15
    cfg.mspr.feedback.max_boost_per_memory = 0.60
    cfg.mspr.feedback.pin_floor_on_boost = True
    obj.config = cfg
    obj._context_graph = build_context_graph(cfg.context_graph)
    obj._context_graph_builder = None
    obj._feedback_store = FeedbackStore(str(tmp_path / "fb.db"))
    obj._success_store = None
    obj._intent_router = None
    obj._role_policy = None
    # Seed one memory node so record_feedback finds something to bump.
    obj._context_graph.backend.upsert_node(
        MemoryNode(id="m1", hash="h", created_at=_now(), updated_at=_now())
    )
    return obj


def test_record_feedback_returns_none_when_disabled(tmp_path):
    obj = _bare_memory(tmp_path)
    obj.config.mspr.feedback.enabled = False
    assert Memory.record_feedback(obj, "m1", helpful=True) is None


def test_record_feedback_bumps_relevance_and_persists(tmp_path):
    obj = _bare_memory(tmp_path)
    result = Memory.record_feedback(obj, "m1", helpful=True, user_id="u1", query="Q?")
    assert result is not None
    assert result["helpful"] is True
    assert result["delta_applied"] == pytest.approx(0.25)
    node = obj._context_graph.backend.get_node("m1")
    assert node.helpful_count == 1
    assert node.last_feedback_at is not None
    assert node.relevance_floor > 0  # floor pinned by helpful bump
    # Persisted in SQLite
    rows = obj._feedback_store.list_events("m1")
    assert len(rows) == 1
    assert rows[0].query == "Q?"
    assert rows[0].query_hash is not None


def test_record_feedback_unhelpful_does_not_pin_floor(tmp_path):
    obj = _bare_memory(tmp_path)
    result = Memory.record_feedback(obj, "m1", helpful=False, user_id="u1")
    assert result["delta_applied"] == pytest.approx(-0.15)
    node = obj._context_graph.backend.get_node("m1")
    assert node.unhelpful_count == 1
    assert node.helpful_count == 0
    assert node.relevance_floor == 0.0  # unchanged


def test_record_feedback_clamps_cumulative_boost(tmp_path):
    obj = _bare_memory(tmp_path)
    # max_boost_per_memory = 0.60; boost_delta = 0.25 → 3rd helpful must
    # clamp to 0.60 - 0.50 = 0.10 (tail end) then zero-out further boosts.
    applied = []
    for _ in range(5):
        applied.append(
            Memory.record_feedback(obj, "m1", helpful=True, user_id="u1")[
                "delta_applied"
            ]
        )
    assert applied[0] == pytest.approx(0.25)
    assert applied[1] == pytest.approx(0.25)
    assert applied[2] == pytest.approx(0.10)
    assert applied[3] == 0.0
    assert applied[4] == 0.0
    stats = obj._feedback_store.aggregate("m1")
    assert stats["net"] == pytest.approx(0.60)


def test_record_feedback_emits_changelog_event(tmp_path):
    obj = _bare_memory(tmp_path)
    # Swap in a facade with a changelog so we can spy on _emit.
    cfg = ContextGraphConfig(
        enabled=True, backend="networkx", log_changes=True,
        persist_path=str(tmp_path / "snap.pkl"),
    )
    obj.config.context_graph = cfg
    obj._context_graph = build_context_graph(cfg)
    obj._context_graph.backend.upsert_node(
        MemoryNode(id="m1", hash="h", created_at=_now(), updated_at=_now())
    )
    spy = _EventSpy()
    obj._context_graph.changelog.subscribe(spy, event_types=["memory_feedback_recorded"])
    Memory.record_feedback(obj, "m1", helpful=True, user_id="u1")
    # Allow the drain thread a moment.
    import time
    for _ in range(20):
        if spy.events:
            break
        time.sleep(0.05)
    assert spy.events, "memory_feedback_recorded event was not delivered"
    ev = spy.events[0]
    assert ev["type"] == "memory_feedback_recorded"
    assert ev["payload"]["helpful"] is True


def test_record_feedback_unknown_memory_returns_none(tmp_path):
    obj = _bare_memory(tmp_path)
    assert Memory.record_feedback(obj, "does-not-exist", helpful=True) is None


def test_reset_feedback_clears_rows(tmp_path):
    obj = _bare_memory(tmp_path)
    Memory.record_feedback(obj, "m1", helpful=True, user_id="u1")
    Memory.record_feedback(obj, "m1", helpful=False, user_id="u2")
    removed = Memory.reset_feedback(obj, user_id="u1")
    assert removed == 1
    remaining = obj._feedback_store.list_events("m1")
    assert len(remaining) == 1
    assert remaining[0].user_id == "u2"
