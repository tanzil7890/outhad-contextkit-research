"""Deterministic replay of the Context-Graph changelog onto a backend.

Every mutation performed by :class:`ContextGraph` is appended to
:class:`ContextChangeLog`. Given the complete, ordered event stream, we
can rebuild an equivalent backend from scratch — which is exactly what
snapshots rely on for point-in-time recovery and audit diffing.

The module is intentionally pure: no network IO, no external services,
no knowledge of ``Memory``. It only needs a changelog and a backend.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from outhad_contextkit.memory.context_graph.backends import (
    ContextGraphBackend,
    _dict_to_edge,
    _dict_to_node,
)
from outhad_contextkit.memory.context_graph.changelog import ContextChangeLog
from outhad_contextkit.memory.context_graph.types import (
    ChangeEvent,
    EdgeType,
    MemoryEdge,
    MemoryNode,
)

logger = logging.getLogger(__name__)


def _apply_event(backend: ContextGraphBackend, ev: ChangeEvent) -> bool:
    """Apply one event; return True on success, False when skipped."""
    payload: Dict[str, Any] = dict(ev.payload or {})
    kind = ev.event_type

    try:
        if kind == "node_added":
            # Early events only carry {"hash": ...} in their payload, so we
            # must stitch in the target_id and any timestamps that happen to
            # be present. Missing fields fall back to safe defaults.
            node_data = {"id": ev.target_id, **payload}
            node_data.setdefault("created_at", ev.timestamp.isoformat())
            node_data.setdefault("updated_at", ev.timestamp.isoformat())
            backend.upsert_node(_dict_to_node(node_data))
            return True

        if kind == "node_updated":
            existing = backend.get_node(ev.target_id)
            if existing is None:
                # Updating before create — treat as additive so we stay
                # deterministic even when the event stream is truncated.
                node_data = {"id": ev.target_id, **payload}
                node_data.setdefault("created_at", ev.timestamp.isoformat())
                node_data.setdefault("updated_at", ev.timestamp.isoformat())
                backend.upsert_node(_dict_to_node(node_data))
                return True
            if "hash" in payload:
                existing.hash = payload["hash"]
            if "version" in payload:
                try:
                    existing.version = int(payload["version"])
                except (TypeError, ValueError):
                    pass
            existing.updated_at = ev.timestamp
            backend.upsert_node(existing)
            return True

        if kind == "node_archived":
            backend.archive_node(ev.target_id)
            return True

        if kind == "node_deleted":
            backend.delete_node(ev.target_id)
            return True

        if kind == "node_relevance_bumped":
            delta = payload.get("delta")
            new_rel = payload.get("new_relevance")
            floor_set = bool(payload.get("floor_set", False))
            if delta is None and new_rel is None:
                return False
            if delta is not None:
                floor_value: Optional[float] = None
                if floor_set:
                    node = backend.get_node(ev.target_id)
                    base = float(node.relevance) if node is not None else 0.0
                    floor_value = max(0.0, min(1.0, base + float(delta)))
                backend.bump_relevance(
                    ev.target_id, float(delta), floor=floor_value
                )
                return True
            # Fall back to absolute-set semantics when only new_relevance
            # survived (e.g. payload was trimmed before persisting).
            node = backend.get_node(ev.target_id)
            if node is None:
                return False
            node.relevance = max(0.0, min(1.0, float(new_rel)))
            if floor_set:
                node.relevance_floor = max(
                    node.relevance_floor, float(new_rel)
                )
            backend.upsert_node(node)
            return True

        if kind in ("edge_added", "edge_decayed"):
            src, dst = _parse_edge_target(ev.target_id)
            if src is None or dst is None:
                return False
            edge_type = EdgeType(
                payload.get("type", EdgeType.REPLY_TO.value)
            )
            weight = float(
                payload.get("weight", 1.0) if payload.get("weight") is not None else 1.0
            )
            backend.upsert_edge(
                MemoryEdge(
                    src=src,
                    dst=dst,
                    type=edge_type,
                    weight=max(0.0, min(1.0, weight)),
                    created_at=ev.timestamp,
                    updated_at=ev.timestamp,
                )
            )
            return True

        if kind == "edge_pruned":
            src, dst = _parse_edge_target(ev.target_id)
            if src is None or dst is None:
                return False
            edge_type = EdgeType(
                payload.get("type", EdgeType.REPLY_TO.value)
            )
            backend.remove_edge(src, dst, edge_type)
            return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("replay: skipping %s on %s (%s)", kind, ev.target_id, exc)
        return False

    return False


def _parse_edge_target(target: str) -> Tuple[Optional[str], Optional[str]]:
    if not target or "->" not in target:
        return None, None
    src, dst = target.split("->", 1)
    return src or None, dst or None


def replay_from_changelog(
    changelog: ContextChangeLog,
    backend: ContextGraphBackend,
    *,
    until: Optional[datetime] = None,
    batch: int = 1000,
) -> Dict[str, int]:
    """Re-apply every changelog event onto ``backend`` in id order.

    ``backend`` should be freshly constructed (empty) to get a faithful
    replica. ``until`` caps the replay at a point-in-time (inclusive).
    Returns ``{"applied": N, "skipped": M}``.
    """
    counts = {"applied": 0, "skipped": 0}
    for ev in changelog.all_events(until=until, batch=batch):
        if _apply_event(backend, ev):
            counts["applied"] += 1
        else:
            counts["skipped"] += 1
    return counts


__all__ = ["replay_from_changelog"]
