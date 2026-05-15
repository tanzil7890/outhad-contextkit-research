"""Phase B — portable JSON-Lines snapshots + changelog replay."""
from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.backends.networkx_backend import (
    NetworkXBackend,
)
from outhad_contextkit.memory.context_graph.changelog import ContextChangeLog
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.context_graph.facade import ContextGraph
from outhad_contextkit.memory.context_graph.replay import replay_from_changelog
from outhad_contextkit.memory.context_graph.types import EdgeType


def _built_cg(tmpdir):
    """Build a CGL facade with changelog persisted under ``tmpdir``."""
    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=True)
    cfg.persist_path = os.path.join(tmpdir, "ctx_graph.pkl")
    cg = build_context_graph(cfg)
    return cfg, cg


def test_export_jsonl_writes_envelope_and_kinds(tmp_path):
    cfg, cg = _built_cg(str(tmp_path))
    cg.upsert_memory_node("m1", "alpha", user_id="u1")
    cg.upsert_memory_node("m2", "beta", user_id="u1")
    cg.upsert_edge("m1", "m2", EdgeType.TOPIC_SIMILAR, weight=0.7)

    path = str(tmp_path / "snap.jsonl")
    written = cg.export_jsonl(path)
    assert written == path
    assert os.path.exists(path)

    with open(path, "r", encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh if l.strip()]

    envelope = lines[0]
    assert envelope["__envelope__"] == 1
    assert envelope["node_count"] == 2
    assert envelope["edge_count"] == 1

    kinds = [entry.get("kind") for entry in lines[1:]]
    assert kinds.count("node") == 2
    assert kinds.count("edge") == 1


def test_import_jsonl_round_trip(tmp_path):
    cfg, cg = _built_cg(str(tmp_path))
    cg.upsert_memory_node("m1", "alpha", user_id="u1")
    cg.upsert_memory_node("m2", "beta", user_id="u1")
    cg.upsert_edge("m1", "m2", EdgeType.REPLY_TO, weight=0.6)

    path = str(tmp_path / "snap.jsonl")
    cg.export_jsonl(path)

    fresh = NetworkXBackend()
    counts = fresh.import_jsonl(path)
    assert counts["nodes"] == 2
    assert counts["edges"] == 1

    # Re-export from the imported backend and compare ignoring the
    # envelope's ``created_at`` field (a free-running timestamp).
    path2 = str(tmp_path / "snap2.jsonl")
    fresh.export_jsonl(path2)

    def _body(p):
        with open(p, "r", encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        env = dict(lines[0])
        env.pop("created_at", None)
        return [env] + lines[1:]

    assert _body(path) == _body(path2)


def test_replay_from_changelog_rebuilds_backend(tmp_path):
    cfg, cg = _built_cg(str(tmp_path))
    cg.upsert_memory_node("m1", "alpha", user_id="u1")
    cg.upsert_memory_node("m2", "beta", user_id="u1")
    cg.upsert_edge("m1", "m2", EdgeType.TOPIC_SIMILAR, weight=0.6)
    cg.bump_relevance("m1", 0.3, set_floor=True)

    # Fresh backend fed by replaying the persistent changelog.
    fresh = NetworkXBackend()
    counts = replay_from_changelog(cg.changelog, fresh)
    assert counts["applied"] >= 3
    assert fresh.get_node("m1") is not None
    assert fresh.get_node("m2") is not None
    assert fresh.edge_count() == 1

    rebuilt = fresh.get_node("m1")
    # Floor should match the one we pinned via bump_relevance.
    assert rebuilt.relevance_floor > 0.0


def test_replay_determinism(tmp_path):
    """Two fresh backends fed the same changelog converge to the same state."""
    cfg, cg = _built_cg(str(tmp_path))
    cg.upsert_memory_node("m1", "alpha", user_id="u1")
    cg.upsert_memory_node("m2", "beta", user_id="u1")
    cg.upsert_edge("m1", "m2", EdgeType.TOPIC_SIMILAR, weight=0.5)
    cg.bump_relevance("m1", 0.2, set_floor=True)

    b1 = NetworkXBackend()
    b2 = NetworkXBackend()
    replay_from_changelog(cg.changelog, b1)
    replay_from_changelog(cg.changelog, b2)

    triples_1 = sorted(
        (n.id, n.hash, round(n.relevance, 6)) for n in b1.iter_nodes()
    )
    triples_2 = sorted(
        (n.id, n.hash, round(n.relevance, 6)) for n in b2.iter_nodes()
    )
    assert triples_1 == triples_2
    assert b1.node_count() == b2.node_count()
    assert b1.edge_count() == b2.edge_count()


def test_replay_until_cutoff(tmp_path):
    cfg, cg = _built_cg(str(tmp_path))
    cg.upsert_memory_node("m1", "alpha", user_id="u1")
    # Ensure a strictly later second event for the cutoff to matter. SQLite
    # stores ISO-8601 timestamps at microsecond resolution; sleep briefly.
    time.sleep(0.02)
    import datetime as _dt

    cutoff = _dt.datetime.utcnow()
    time.sleep(0.02)
    cg.upsert_memory_node("m2", "beta", user_id="u1")

    fresh = NetworkXBackend()
    replay_from_changelog(cg.changelog, fresh, until=cutoff)
    assert fresh.get_node("m1") is not None
    assert fresh.get_node("m2") is None


def test_memory_snapshot_requires_cgl_enabled():
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    obj._context_graph = None
    with pytest.raises(RuntimeError):
        Memory.snapshot_context_graph(obj, "/tmp/ignored.jsonl")
    with pytest.raises(RuntimeError):
        Memory.replay_context_graph(obj)


def test_memory_replay_requires_changelog(tmp_path):
    from outhad_contextkit.memory.main import Memory

    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=False)
    cg = build_context_graph(cfg)
    obj = object.__new__(Memory)
    obj._context_graph = cg
    with pytest.raises(RuntimeError):
        Memory.replay_context_graph(obj)


def test_memory_snapshot_and_replay_end_to_end(tmp_path):
    from outhad_contextkit.memory.main import Memory

    cfg = ContextGraphConfig(enabled=True, backend="networkx", log_changes=True)
    cfg.persist_path = str(tmp_path / "graph.pkl")
    cg = build_context_graph(cfg)
    cg.upsert_memory_node("m1", "alpha", user_id="u1")
    cg.upsert_memory_node("m2", "beta", user_id="u1")
    cg.upsert_edge("m1", "m2", EdgeType.TOPIC_SIMILAR, weight=0.5)

    obj = object.__new__(Memory)
    obj._context_graph = cg

    out_path = str(tmp_path / "snap.jsonl")
    assert Memory.snapshot_context_graph(obj, out_path) == out_path

    replayed = Memory.replay_context_graph(obj)
    assert isinstance(replayed, ContextGraph)
    assert replayed.stats()["nodes"] == 2
    assert replayed.stats()["edges"] == 1


def test_unknown_event_type_is_skipped(tmp_path):
    """Forward-compat: future event types must be counted but not raise."""
    changelog = ContextChangeLog(str(tmp_path / "changes.db"))
    from outhad_contextkit.memory.context_graph.types import ChangeEvent
    import datetime as _dt

    changelog.append(
        ChangeEvent(
            event_type="future_event",
            target_id="anything",
            timestamp=_dt.datetime.utcnow(),
            user_id=None,
            payload={},
        )
    )
    backend = NetworkXBackend()
    counts = replay_from_changelog(changelog, backend)
    assert counts["skipped"] == 1
    assert counts["applied"] == 0
