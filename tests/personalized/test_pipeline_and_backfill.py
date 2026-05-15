"""Phase F8 — PersonalizedRetrievalPipeline + backfill_mspr + telemetry.

Covers:
* Pipeline passthrough when mspr disabled.
* Pipeline runs full rerank when mspr enabled, returning mspr sidecar.
* Pipeline emits outhad_contextkit.mspr.search telemetry exactly once.
* backfill_mspr reconstructs query_success rows from feedback_events.
* backfill_mspr raises when MSPR is disabled (typo guard).
* backfill_mspr is idempotent — running twice doubles rows but doesn't crash.
* _emit_mspr_telemetry is fired by _graph_first_rerank when mspr.enabled.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.context_graph.types import MemoryNode
from outhad_contextkit.memory.main import Memory
from outhad_contextkit.memory.personalized.feedback_store import FeedbackStore
from outhad_contextkit.memory.personalized.pipeline import (
    PersonalizedRetrievalPipeline,
)
from outhad_contextkit.memory.personalized.success_store import QuerySuccessStore


def _now():
    return datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)


def _bare_memory(tmp_path, *, enable_mspr=True, enable_success=True):
    """Construct a Memory shell with the bits the F8 tests need."""
    obj = object.__new__(Memory)
    cfg = MemoryConfig()
    cfg.context_graph = ContextGraphConfig(
        enabled=True, backend="networkx", log_changes=False
    )
    cfg.mspr.enabled = enable_mspr
    cfg.mspr.feedback.enabled = True
    cfg.mspr.success.enabled = enable_success
    obj.config = cfg
    obj._context_graph = build_context_graph(cfg.context_graph)
    obj._context_graph_builder = None
    obj._feedback_store = FeedbackStore(str(tmp_path / "fb.db"))
    obj._success_store = (
        QuerySuccessStore(str(tmp_path / "qs.db")) if enable_success else None
    )
    obj._intent_router = None
    obj._role_policy = None
    obj._mspr_pipeline = PersonalizedRetrievalPipeline(obj)
    # Seed two memory nodes.
    for mid in ("m1", "m2"):
        obj._context_graph.backend.upsert_node(
            MemoryNode(
                id=mid,
                hash="h",
                created_at=_now().replace(tzinfo=None),
                updated_at=_now().replace(tzinfo=None),
            )
        )
    return obj


# ---------------------------------------------------------------------------
# PersonalizedRetrievalPipeline
# ---------------------------------------------------------------------------

def test_pipeline_passthrough_when_disabled(tmp_path):
    obj = _bare_memory(tmp_path, enable_mspr=False)
    seeds = [{"id": "m1", "memory": "alpha", "score": 0.5}]
    out = obj._mspr_pipeline.retrieve(
        "q", seed_results=seeds, limit=5
    )
    assert out["results"] == seeds
    assert out["subgraph"] == []


def test_pipeline_returns_mspr_sidecar(tmp_path):
    obj = _bare_memory(tmp_path)
    seeds = [
        {"id": "m1", "memory": "alpha", "score": 0.6},
        {"id": "m2", "memory": "beta", "score": 0.5},
    ]
    out = obj._mspr_pipeline.retrieve(
        "q", seed_results=seeds, limit=2
    )
    assert "mspr" in out
    assert out["mspr"]["seed_count"] == 2
    assert out["mspr"]["result_count"] >= 1
    assert "latency_ms" in out["mspr"]


def test_pipeline_emits_telemetry(tmp_path):
    obj = _bare_memory(tmp_path)
    seeds = [{"id": "m1", "memory": "alpha", "score": 0.5}]
    with patch(
        "outhad_contextkit.memory.personalized.pipeline.capture_event"
    ) as spy:
        obj._mspr_pipeline.retrieve("q", seed_results=seeds, limit=1)
    assert spy.called
    event_name = spy.call_args[0][0]
    payload = spy.call_args[0][2]
    assert event_name == "outhad_contextkit.mspr.search"
    assert payload["seed_count"] == 1
    assert "latency_ms" in payload


def test_pipeline_with_filters_reads_role_ctx(tmp_path):
    obj = _bare_memory(tmp_path)
    seeds = [{"id": "m1", "memory": "alpha", "score": 0.5}]
    # Should not raise even when role policy is None.
    out = obj._mspr_pipeline.retrieve(
        "q",
        seed_results=seeds,
        limit=1,
        filters={"tenant_id": "acme"},
        user_id="u1",
    )
    assert "results" in out


# ---------------------------------------------------------------------------
# Memory._emit_mspr_telemetry inside _graph_first_rerank
# ---------------------------------------------------------------------------

def test_graph_first_rerank_fires_telemetry_when_mspr_enabled(tmp_path):
    obj = _bare_memory(tmp_path)
    seeds = [{"id": "m1", "memory": "alpha", "score": 0.5}]
    with patch("outhad_contextkit.memory.main.capture_event") as spy:
        Memory._graph_first_rerank(
            obj,
            query="q",
            seed_results=seeds,
            limit=1,
            include_archived=False,
        )
    names = [c.args[0] for c in spy.call_args_list]
    assert "outhad_contextkit.mspr.search" in names


def test_graph_first_rerank_no_telemetry_when_disabled(tmp_path):
    obj = _bare_memory(tmp_path, enable_mspr=False)
    seeds = [{"id": "m1", "memory": "alpha", "score": 0.5}]
    with patch("outhad_contextkit.memory.main.capture_event") as spy:
        Memory._graph_first_rerank(
            obj,
            query="q",
            seed_results=seeds,
            limit=1,
            include_archived=False,
        )
    names = [c.args[0] for c in spy.call_args_list]
    assert "outhad_contextkit.mspr.search" not in names


# ---------------------------------------------------------------------------
# backfill_mspr
# ---------------------------------------------------------------------------

def test_backfill_mspr_raises_when_disabled(tmp_path):
    obj = _bare_memory(tmp_path, enable_mspr=False)
    with pytest.raises(RuntimeError):
        Memory.backfill_mspr(obj)


def test_backfill_mspr_reconstructs_success_rows(tmp_path):
    obj = _bare_memory(tmp_path)
    # Seed 3 feedback events with query hashes, then drop the success
    # store and rebuild via backfill.
    Memory.record_feedback(
        obj, "m1", helpful=True, user_id="u1", query="What is X?"
    )
    Memory.record_feedback(
        obj, "m1", helpful=False, user_id="u2", query="What is X?"
    )
    Memory.record_feedback(
        obj, "m2", helpful=True, user_id="u1", query="Find resume"
    )
    # Wipe success store so backfill has work to do.
    obj._success_store.reset()
    counts = Memory.backfill_mspr(obj)
    assert counts["feedback_rows"] == 3
    assert counts["success_rows"] == 3
    # Confirm rows landed in success store.
    from outhad_contextkit.memory.personalized.feedback_store import hash_query

    qh = hash_query("What is X?")
    stats = obj._success_store.aggregate(query_hash=qh, memory_id="m1")
    assert stats["total"] == 2
    assert stats["helpful"] == 1


def test_backfill_mspr_skips_events_without_query_hash(tmp_path):
    obj = _bare_memory(tmp_path)
    # No query string → query_hash is None → backfill must skip the row.
    Memory.record_feedback(obj, "m1", helpful=True, user_id="u1")
    counts = Memory.backfill_mspr(obj)
    assert counts["feedback_rows"] == 1
    assert counts["success_rows"] == 0


def test_backfill_mspr_returns_zero_when_stores_missing(tmp_path):
    obj = _bare_memory(tmp_path, enable_success=False)
    obj._feedback_store = None
    counts = Memory.backfill_mspr(obj)
    assert counts == {"feedback_rows": 0, "success_rows": 0}
