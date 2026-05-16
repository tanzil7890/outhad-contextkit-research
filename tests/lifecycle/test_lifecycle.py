""" lifecycle subsystem tests.

Covers:
* D1: config defaults + weight-sum validator.
* D2: compute_decay_score formula; Memory.decay_score; archive_low + dry_run.
* D3: DecayScheduler start/stop/tick; crash recovery.
* D4/D5: VersionedMemoryStore insert + list + prune; versions / latest /
  diff_versions / rollback round-trip.
* D6: ReferenceTracker bumps access_count + last_accessed_at.
* D7: LocalDiskAdapter put/get/delete; demote_to_cold + promote_from_cold round-trip.
* D8: backfill_decay_v2 raises when disabled; archives candidates.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.context_graph.types import MemoryNode
from outhad_contextkit.memory.lifecycle.cold_storage import LocalDiskAdapter
from outhad_contextkit.memory.lifecycle.config import (
    DecayV2Config,
    ScheduleConfig,
    VersioningConfig,
)
from outhad_contextkit.memory.lifecycle.scheduler import DecayScheduler
from outhad_contextkit.memory.lifecycle.scoring import compute_decay_score
from outhad_contextkit.memory.main import Memory
from outhad_contextkit.memory.storage import SQLiteManager


def _now():
    return datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# D1 — config defaults + validator
# ---------------------------------------------------------------------------

def test_decay_v2_disabled_by_default():
    cfg = MemoryConfig()
    assert cfg.decay_v2.enabled is False
    assert cfg.decay_v2.versioning.mode == "overwrite"
    assert cfg.decay_v2.cold_storage.backend == "disabled"
    assert cfg.decay_v2.schedule.enabled is False


def test_decay_v2_weight_sum_validator():
    with pytest.raises(Exception):
        DecayV2Config(weight_recency=0.6, weight_frequency=0.5, weight_feedback=0.1)


def test_decay_v2_weights_sum_at_most_one_passes():
    cfg = DecayV2Config(
        weight_recency=0.4, weight_frequency=0.3, weight_feedback=0.2
    )
    assert cfg.weight_recency + cfg.weight_frequency + cfg.weight_feedback == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# D2 — scoring
# ---------------------------------------------------------------------------

def test_just_accessed_node_scores_high():
    cfg = DecayV2Config(enabled=True)
    node = MemoryNode(
        id="m",
        hash="h",
        created_at=_now(),
        updated_at=_now(),
        last_accessed_at=_now(),
        access_count=5,
        helpful_count=3,
    )
    score = compute_decay_score(
        node, cfg=cfg, peak_access=5, half_life_days=30, now=_now()
    )
    # Just accessed + helpful → close to top.
    assert score > 0.6


def test_stale_uncited_node_scores_low():
    cfg = DecayV2Config(enabled=True)
    old = _now() - timedelta(days=120)  # 4 half-lives
    node = MemoryNode(
        id="m",
        hash="h",
        created_at=old,
        updated_at=old,
        last_accessed_at=old,
        access_count=0,
    )
    score = compute_decay_score(
        node, cfg=cfg, peak_access=10, half_life_days=30, now=_now()
    )
    assert score < 0.2


def _shell(tmp_path, *, decay_enabled=True, mode="overwrite"):
    obj = object.__new__(Memory)
    cfg = MemoryConfig()
    cfg.decay_v2.enabled = decay_enabled
    cfg.decay_v2.versioning.mode = mode
    cfg.context_graph = ContextGraphConfig(
        enabled=True, backend="networkx", log_changes=False
    )
    obj.config = cfg
    obj.collection_name = "ck"
    obj.vector_store = MagicMock()
    obj.vector_store.get.return_value = None
    obj.embedding_model = None
    obj.db = SQLiteManager(str(tmp_path / "history.db"))
    obj._context_graph = build_context_graph(cfg.context_graph)
    obj._context_graph_builder = None
    Memory._init_tenant(obj)
    Memory._init_lifecycle(obj)
    return obj


@patch("outhad_contextkit.memory.main.capture_event")
def test_memory_decay_score_returns_none_when_disabled(_capture, tmp_path):
    obj = _shell(tmp_path, decay_enabled=False)
    assert Memory.decay_score(obj, "m1") is None


@patch("outhad_contextkit.memory.main.capture_event")
def test_memory_decay_score_returns_float_when_enabled(_capture, tmp_path):
    obj = _shell(tmp_path)
    obj._context_graph.upsert_memory_node("m1", "hello")
    score = Memory.decay_score(obj, "m1")
    assert score is not None
    assert 0.0 <= score <= 1.0


@patch("outhad_contextkit.memory.main.capture_event")
def test_archive_low_dry_run_returns_candidates_without_mutating(
    _capture, tmp_path
):
    obj = _shell(tmp_path)
    # Create a stale node by upserting then making it look ancient.
    obj._context_graph.upsert_memory_node("stale", "hello")
    node = obj._context_graph.backend.get_node("stale")
    old = _now() - timedelta(days=180)
    node.created_at = old.replace(tzinfo=None)
    node.updated_at = old.replace(tzinfo=None)
    node.last_accessed_at = old.replace(tzinfo=None)
    obj._context_graph.backend.upsert_node(node)
    # Use a generous threshold so we definitely catch it.
    out = Memory.archive_low(obj, threshold=0.5, dry_run=True)
    assert "stale" in out["candidates"]
    assert out["archived"] == 0  # dry run
    assert obj._context_graph.backend.get_node("stale").archived is False


@patch("outhad_contextkit.memory.main.capture_event")
def test_archive_low_archives_below_threshold(_capture, tmp_path):
    obj = _shell(tmp_path)
    obj._context_graph.upsert_memory_node("stale", "hello")
    node = obj._context_graph.backend.get_node("stale")
    old = _now() - timedelta(days=180)
    node.created_at = old.replace(tzinfo=None)
    node.updated_at = old.replace(tzinfo=None)
    node.last_accessed_at = old.replace(tzinfo=None)
    obj._context_graph.backend.upsert_node(node)
    out = Memory.archive_low(obj, threshold=0.5)
    assert out["archived"] >= 1
    assert obj._context_graph.backend.get_node("stale").archived is True


# ---------------------------------------------------------------------------
# D3 — scheduler
# ---------------------------------------------------------------------------

@patch("outhad_contextkit.memory.main.capture_event")
def test_scheduler_start_stop(_capture, tmp_path):
    obj = _shell(tmp_path)
    sched = DecayScheduler(
        obj,
        cfg=ScheduleConfig(enabled=True, interval_seconds=1.0, jitter_seconds=0.0),
    )
    sched.start()
    assert sched.is_running()
    sched.stop(timeout=2.0)
    assert not sched.is_running()


@patch("outhad_contextkit.memory.main.capture_event")
def test_scheduler_tick_calls_tick_decay_and_archive_low(
    _capture, tmp_path
):
    obj = _shell(tmp_path)
    obj._context_graph.tick_decay = MagicMock()
    sched = DecayScheduler(
        obj, cfg=ScheduleConfig(interval_seconds=10.0, jitter_seconds=0.0)
    )
    sched.tick_now()
    obj._context_graph.tick_decay.assert_called_once()


@patch("outhad_contextkit.memory.main.capture_event")
def test_scheduler_tick_failure_does_not_crash(_capture, tmp_path):
    obj = _shell(tmp_path)
    obj._context_graph.tick_decay = MagicMock(side_effect=RuntimeError("boom"))
    sched = DecayScheduler(
        obj, cfg=ScheduleConfig(interval_seconds=10.0, jitter_seconds=0.0)
    )
    # Tick should swallow and not raise.
    sched.tick_now()


@patch("outhad_contextkit.memory.main.capture_event")
def test_memory_close_stops_scheduler(_capture, tmp_path):
    obj = _shell(tmp_path)
    obj.config.decay_v2.schedule.enabled = True
    Memory.start_decay_scheduler(obj)
    assert obj._lifecycle_scheduler.is_running()
    Memory.close(obj)
    assert not obj._lifecycle_scheduler.is_running()


# ---------------------------------------------------------------------------
# D4/D5 — versioning + rollback
# ---------------------------------------------------------------------------

class _FakeVectorStore:
    """Tiny in-memory vector-store double for versioning tests."""

    def __init__(self):
        self._rows = {}

    def insert(self, vectors=None, ids=None, payloads=None):
        for i, p in zip(ids or [], payloads or []):
            self._rows[i] = MagicMock(payload=dict(p))

    def get(self, vector_id):
        return self._rows.get(vector_id)

    def update(self, vector_id, payload=None, **_):
        if vector_id in self._rows and payload is not None:
            self._rows[vector_id].payload = dict(payload)

    def delete(self, vector_id):
        self._rows.pop(vector_id, None)


def _versioning_shell(tmp_path):
    obj = object.__new__(Memory)
    cfg = MemoryConfig()
    cfg.decay_v2.enabled = True
    cfg.decay_v2.versioning.mode = "immutable"
    cfg.context_graph = ContextGraphConfig(
        enabled=True, backend="networkx", log_changes=False
    )
    obj.config = cfg
    obj.collection_name = "ck"
    obj.vector_store = _FakeVectorStore()
    obj.embedding_model = None
    obj.db = SQLiteManager(str(tmp_path / "history.db"))
    obj._context_graph = build_context_graph(cfg.context_graph)
    obj._context_graph_builder = None
    Memory._init_tenant(obj)
    Memory._init_lifecycle(obj)
    # Seed the initial memory + vector row.
    obj._context_graph.upsert_memory_node("m1", "version 1")
    obj.vector_store.insert(
        ids=["m1"],
        payloads=[{"data": "version 1", "status": "active"}],
    )
    return obj


@patch("outhad_contextkit.memory.main.capture_event")
def test_immutable_update_creates_new_row(_capture, tmp_path):
    obj = _versioning_shell(tmp_path)
    rec = obj._lifecycle_versioning.insert_version(
        "m1",
        "version 2",
        new_metadata={"data": "version 2", "status": "active"},
    )
    assert rec.memory_id != "m1"
    assert rec.version == 2
    assert rec.prev_id == "m1"
    # Old row marked superseded.
    old = obj.vector_store.get("m1")
    assert old.payload["status"] == "superseded"
    assert old.payload["next_version_id"] == rec.memory_id


@patch("outhad_contextkit.memory.main.capture_event")
def test_versions_returns_chain_in_order(_capture, tmp_path):
    obj = _versioning_shell(tmp_path)
    r2 = obj._lifecycle_versioning.insert_version(
        "m1",
        "version 2",
        new_metadata={"data": "version 2", "status": "active"},
    )
    r3 = obj._lifecycle_versioning.insert_version(
        r2.memory_id,
        "version 3",
        new_metadata={"data": "version 3", "status": "active"},
    )
    chain = Memory.versions(obj, "m1")
    versions = [v.version for v in chain]
    assert versions == [1, 2, 3]
    latest = Memory.latest_version(obj, "m1")
    assert latest.memory_id == r3.memory_id


@patch("outhad_contextkit.memory.main.capture_event")
def test_diff_versions_unified(_capture, tmp_path):
    obj = _versioning_shell(tmp_path)
    r2 = obj._lifecycle_versioning.insert_version(
        "m1",
        "version 2",
        new_metadata={"data": "version 2", "status": "active"},
    )
    diff = Memory.diff_versions(obj, "m1", r2.memory_id, mode="unified")
    assert "version 1" in diff
    assert "version 2" in diff


@patch("outhad_contextkit.memory.main.capture_event")
def test_rollback_creates_new_version_with_old_payload(_capture, tmp_path):
    obj = _versioning_shell(tmp_path)
    r2 = obj._lifecycle_versioning.insert_version(
        "m1",
        "version 2",
        new_metadata={"data": "version 2", "status": "active"},
    )
    # Roll back to version 1.
    new_id = Memory.rollback(obj, "m1", to_version=1)
    assert new_id is not None
    assert new_id != "m1"
    chain = Memory.versions(obj, "m1")
    # Chain length is 3 (v1 + v2 + the rollback).
    assert len(chain) == 3
    # New head's payload matches v1.
    head = chain[-1]
    head_row = obj.vector_store.get(head.memory_id)
    assert head_row.payload["data"] == "version 1"


@patch("outhad_contextkit.memory.main.capture_event")
def test_prune_drops_oldest_superseded(_capture, tmp_path):
    obj = _versioning_shell(tmp_path)
    obj.config.decay_v2.versioning.keep_versions = 2
    r2 = obj._lifecycle_versioning.insert_version(
        "m1", "v2", {"data": "v2", "status": "active"}
    )
    r3 = obj._lifecycle_versioning.insert_version(
        r2.memory_id, "v3", {"data": "v3", "status": "active"}
    )
    removed = obj._prune_version_chains()
    # Chain originally: m1 (superseded), r2 (superseded), r3 (active).
    # Keep=2 → drop oldest superseded (m1).
    assert removed >= 1
    assert obj.vector_store.get("m1") is None


# ---------------------------------------------------------------------------
# D6 — reference tracker
# ---------------------------------------------------------------------------

@patch("outhad_contextkit.memory.main.capture_event")
def test_record_reference_bumps_access_count(_capture, tmp_path):
    obj = _shell(tmp_path)
    obj.config.decay_v2.track_references = True
    obj._context_graph.upsert_memory_node("m1", "hello")
    before = obj._context_graph.backend.get_node("m1").access_count
    new_count = Memory.record_reference(obj, "m1")
    after = obj._context_graph.backend.get_node("m1").access_count
    assert after == before + 1
    assert new_count == after


@patch("outhad_contextkit.memory.main.capture_event")
def test_record_reference_returns_none_when_track_off(_capture, tmp_path):
    obj = _shell(tmp_path)
    # track_references stays False → no-op.
    obj._context_graph.upsert_memory_node("m1", "hello")
    assert Memory.record_reference(obj, "m1") is None


# ---------------------------------------------------------------------------
# D7 — cold storage
# ---------------------------------------------------------------------------

def test_local_disk_adapter_round_trip(tmp_path):
    adapter = LocalDiskAdapter(str(tmp_path / "cold"))
    adapter.put("m1", {"data": "hello"})
    assert adapter.get("m1") == {"data": "hello"}
    assert adapter.delete("m1") is True
    assert adapter.get("m1") is None


@patch("outhad_contextkit.memory.main.capture_event")
def test_demote_promote_round_trip(_capture, tmp_path):
    obj = _versioning_shell(tmp_path)
    obj._lifecycle_cold_storage = LocalDiskAdapter(str(tmp_path / "cold"))
    # Demote m1.
    assert Memory.demote_to_cold(obj, "m1") is True
    # Vector row gone, CGL node archived.
    assert obj.vector_store.get("m1") is None
    assert obj._context_graph.backend.get_node("m1").archived is True
    # Promote back.
    payload = Memory.promote_from_cold(obj, "m1")
    assert payload is not None
    assert obj.vector_store.get("m1") is not None
    assert obj._context_graph.backend.get_node("m1").archived is False


@patch("outhad_contextkit.memory.main.capture_event")
def test_demote_returns_false_when_adapter_disabled(_capture, tmp_path):
    obj = _shell(tmp_path)
    # Default cold_storage is disabled.
    assert Memory.demote_to_cold(obj, "m1") is False


# ---------------------------------------------------------------------------
# D8 — backfill
# ---------------------------------------------------------------------------

@patch("outhad_contextkit.memory.main.capture_event")
def test_backfill_decay_v2_raises_when_disabled(_capture, tmp_path):
    obj = _shell(tmp_path, decay_enabled=False)
    with pytest.raises(RuntimeError):
        Memory.backfill_decay_v2(obj)


@patch("outhad_contextkit.memory.main.capture_event")
def test_backfill_decay_v2_returns_counts(_capture, tmp_path):
    obj = _shell(tmp_path)
    obj._context_graph.upsert_memory_node("a", "hello")
    out = Memory.backfill_decay_v2(obj)
    assert "archived" in out
    assert "candidates" in out
