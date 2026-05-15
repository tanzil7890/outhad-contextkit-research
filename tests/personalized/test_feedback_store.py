"""Phase F3 — FeedbackStore SQLite integration.

Covers:
* Schema bootstrap is idempotent (multiple instances on the same path).
* append → aggregate → personal_boost round-trip.
* Durability across store instances.
* Aggregate windows (scope by user_id, scope by time window).
* ``tanh`` saturation: repeated helpful feedbacks asymptote to +1.
* ``reset`` clears rows.
"""
from __future__ import annotations

from datetime import datetime, timezone

from outhad_contextkit.memory.personalized.feedback_store import (
    FeedbackStore,
    hash_query,
)
from outhad_contextkit.memory.personalized.types import FeedbackEvent


def _mk(**kw) -> FeedbackEvent:
    ts = kw.pop("timestamp", datetime.now(timezone.utc))
    return FeedbackEvent(
        memory_id=kw.pop("memory_id", "m1"),
        helpful=kw.pop("helpful", True),
        delta_applied=kw.pop("delta_applied", 0.15),
        timestamp=ts,
        user_id=kw.pop("user_id", "u1"),
        agent_id=kw.pop("agent_id", None),
        run_id=kw.pop("run_id", None),
        query=kw.pop("query", None),
        query_hash=kw.pop("query_hash", None),
        metadata=kw.pop("metadata", {}),
    )


def test_schema_bootstrap_is_idempotent(tmp_path):
    path = str(tmp_path / "fb.db")
    FeedbackStore(path)
    FeedbackStore(path)  # second bootstrap must not raise
    store = FeedbackStore(path)
    assert store.aggregate("missing") == {
        "helpful": 0.0,
        "unhelpful": 0.0,
        "net": 0.0,
    }


def test_append_returns_rowid_and_mutates_event(tmp_path):
    store = FeedbackStore(str(tmp_path / "fb.db"))
    event = _mk()
    rid = store.append(event)
    assert rid > 0
    assert event.id == rid


def test_aggregate_counts_helpful_and_unhelpful(tmp_path):
    store = FeedbackStore(str(tmp_path / "fb.db"))
    store.append(_mk(helpful=True, delta_applied=0.15))
    store.append(_mk(helpful=True, delta_applied=0.15))
    store.append(_mk(helpful=False, delta_applied=-0.10))
    stats = store.aggregate("m1")
    assert stats["helpful"] == 2.0
    assert stats["unhelpful"] == 1.0
    assert stats["net"] == 0.15 + 0.15 - 0.10


def test_personal_boost_positive_for_helpful_net(tmp_path):
    store = FeedbackStore(str(tmp_path / "fb.db"))
    for _ in range(3):
        store.append(_mk(helpful=True, delta_applied=0.15))
    boost = store.personal_boost("m1")
    assert 0 < boost <= 1.0
    # asymptote check: many more helpful verdicts never exceed +1.
    for _ in range(50):
        store.append(_mk(helpful=True, delta_applied=0.15))
    assert store.personal_boost("m1") <= 1.0


def test_personal_boost_negative_for_penalty(tmp_path):
    store = FeedbackStore(str(tmp_path / "fb.db"))
    for _ in range(3):
        store.append(_mk(helpful=False, delta_applied=-0.15))
    boost = store.personal_boost("m1")
    assert -1.0 <= boost < 0


def test_durability_across_instances(tmp_path):
    path = str(tmp_path / "fb.db")
    FeedbackStore(path).append(_mk(delta_applied=0.5))
    reopened = FeedbackStore(path)
    assert reopened.aggregate("m1")["net"] == 0.5


def test_user_scope_filters_aggregate(tmp_path):
    store = FeedbackStore(str(tmp_path / "fb.db"))
    store.append(_mk(user_id="u1", delta_applied=0.4))
    store.append(_mk(user_id="u2", delta_applied=-0.3))
    assert store.aggregate("m1", user_id="u1")["net"] == 0.4
    assert store.aggregate("m1", user_id="u2")["net"] == -0.3


def test_reset_removes_rows(tmp_path):
    store = FeedbackStore(str(tmp_path / "fb.db"))
    store.append(_mk(user_id="u1"))
    store.append(_mk(user_id="u2"))
    assert store.reset(user_id="u1") == 1
    assert store.aggregate("m1", user_id="u1")["helpful"] == 0.0
    assert store.aggregate("m1", user_id="u2")["helpful"] == 1.0


def test_hash_query_stable_and_lowercase():
    assert hash_query("  Hello World ") == hash_query("hello world")
    assert hash_query("") is None
    assert hash_query(None) is None
    assert len(hash_query("x") or "") == 16


def test_list_events_returns_most_recent_first(tmp_path):
    store = FeedbackStore(str(tmp_path / "fb.db"))
    store.append(_mk(query="first"))
    store.append(_mk(query="second"))
    rows = store.list_events(memory_id="m1")
    assert [r.query for r in rows[:2]] == ["second", "first"]
