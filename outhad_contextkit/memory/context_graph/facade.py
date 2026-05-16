"""Public facade for the Context-Graph Layer.

:class:`ContextGraph` owns the backend, the change timeline, and the
housekeeping primitives (decay/prune). It is consumed by
:class:`~outhad_contextkit.memory.context_graph.builder.IncrementalGraphBuilder`
and by the graph-first retrieval pipeline.

The facade is intentionally side-effect free unless the CGL is enabled; any
no-op path returns empty collections so callers can treat it as always-on.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from outhad_contextkit.memory.context_graph.backends import ContextGraphBackend
from outhad_contextkit.memory.context_graph.changelog import ContextChangeLog
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.context_graph.types import (
    ChangeEvent,
    EdgeType,
    MemoryEdge,
    MemoryNode,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.utcnow()


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class ContextGraph:
    """High-level orchestrator for the memory-as-node graph."""

    def __init__(
        self,
        config: ContextGraphConfig,
        backend: ContextGraphBackend,
        changelog: Optional[ContextChangeLog] = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.changelog = changelog
        self._lock = threading.RLock()
        self._last_decay_tick: Optional[datetime] = None

        if config.persist_path and os.path.exists(config.persist_path):
            try:
                self.backend.load(config.persist_path)
            except Exception as exc:  # pragma: no cover - load is best-effort
                logger.warning("Context graph failed to load snapshot: %s", exc)

    # ---- public helpers ------------------------------------------------
    @staticmethod
    def hash_text(text: str) -> str:
        return _hash_text(text)

    # ---- node lifecycle ------------------------------------------------
    def upsert_memory_node(
        self,
        memory_id: str,
        text: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ) -> MemoryNode:
        now = _utcnow()
        #  fall back to metadata-supplied tenant tags so
        # callers that already pass them via the payload metadata get
        # graph-level partitioning for free.
        meta_dict = dict(metadata or {})
        if tenant_id is None:
            tenant_id = meta_dict.get("tenant_id")
        if sub_tenant_id is None:
            sub_tenant_id = meta_dict.get("sub_tenant_id")
        existing = self.backend.get_node(memory_id)
        if existing is None:
            node = MemoryNode(
                id=memory_id,
                hash=_hash_text(text),
                created_at=now,
                updated_at=now,
                version=1,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                tenant_id=tenant_id,
                sub_tenant_id=sub_tenant_id,
                metadata=meta_dict,
            )
            self.backend.upsert_node(node)
            self._emit("node_added", node.id, user_id, {"hash": node.hash})
        else:
            new_hash = _hash_text(text)
            existing.updated_at = now
            if new_hash != existing.hash:
                existing.prev_version_id = existing.id
                existing.version += 1
                existing.hash = new_hash
                self._emit(
                    "node_updated",
                    existing.id,
                    user_id,
                    {"version": existing.version, "hash": existing.hash},
                )
            if metadata:
                existing.metadata.update(metadata)
            # Stamp tenant fields if the call carried them or the
            # node was previously NULL — never overwrite a non-NULL
            # tenant with a NULL value (defensive).
            if tenant_id is not None:
                existing.tenant_id = tenant_id
            if sub_tenant_id is not None:
                existing.sub_tenant_id = sub_tenant_id
            self.backend.upsert_node(existing)
            node = existing

        if embedding is not None:
            try:
                self.backend.set_embedding(memory_id, list(embedding))
            except Exception as exc:  # pragma: no cover - backend dependent
                logger.debug("Backend rejected embedding for %s: %s", memory_id, exc)
        return node

    def archive_memory_node(self, memory_id: str, user_id: Optional[str] = None) -> None:
        self.backend.archive_node(memory_id)
        self._emit("node_archived", memory_id, user_id, {})

    def bump_relevance(
        self,
        memory_id: str,
        delta: float,
        *,
        set_floor: bool = False,
    ) -> Optional[float]:
        """Nudge ``relevance`` by ``delta`` and emit a change event.

        When ``set_floor=True`` and ``delta > 0`` the post-bump relevance is
        also pinned as a floor so subsequent :meth:`tick_decay` cycles cannot
        push the node below it. Returns the new relevance, or ``None`` when
        the CGL is disabled or the node is unknown.
        """
        if not self.config.enabled:
            return None
        with self._lock:
            node = self.backend.get_node(memory_id)
            if node is None:
                return None
            delta_f = float(delta)
            floor: Optional[float] = None
            if set_floor and delta_f > 0:
                floor = max(0.0, min(1.0, float(node.relevance) + delta_f))
            new_rel = self.backend.bump_relevance(memory_id, delta_f, floor=floor)
            if new_rel is None:
                return None
            self._emit(
                "node_relevance_bumped",
                memory_id,
                node.user_id,
                {
                    "delta": delta_f,
                    "new_relevance": float(new_rel),
                    "floor_set": bool(set_floor and delta_f > 0),
                },
            )
            return new_rel

    def delete_memory_node(self, memory_id: str, user_id: Optional[str] = None) -> None:
        self.backend.delete_node(memory_id)
        self._emit("node_deleted", memory_id, user_id, {})

    # ---- frequency tracking  --------------------------------
    def record_access(
        self,
        memory_id: str,
        *,
        delta: int = 1,
        user_id: Optional[str] = None,
    ) -> Optional[int]:
        """Increment the ``access_count`` counter for ``memory_id``.

        Returns the new count, or ``None`` when the CGL is disabled or the
        node is unknown. Emits a ``node_accessed`` change-bus event so
        subscribers can track popularity in real time.
        """
        if not self.config.enabled:
            return None
        with self._lock:
            new_count = self.backend.bump_access_count(memory_id, delta=int(delta))
            if new_count is None:
                return None
            self._emit(
                "node_accessed",
                memory_id,
                user_id,
                {"delta": int(delta), "access_count": int(new_count)},
            )
            return int(new_count)

    def max_access_count(self) -> int:
        """Return the max ``access_count`` over all live nodes."""
        if not self.config.enabled:
            return 0
        try:
            return int(self.backend.max_access_count())
        except Exception:  # pragma: no cover - backend dependent
            return 0

    # ---- edge lifecycle ------------------------------------------------
    def upsert_edge(
        self,
        src: str,
        dst: str,
        edge_type: EdgeType,
        *,
        weight: float = 1.0,
        evidence: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> MemoryEdge:
        now = _utcnow()
        edge = MemoryEdge(
            src=src,
            dst=dst,
            type=edge_type,
            weight=max(0.0, min(1.0, float(weight))),
            created_at=now,
            updated_at=now,
            evidence=evidence,
            metadata=dict(metadata or {}),
        )
        self.backend.upsert_edge(edge)
        self._emit(
            "edge_added",
            f"{src}->{dst}",
            user_id,
            {"type": edge_type.value, "weight": edge.weight},
        )
        return edge

    def remove_edge(
        self,
        src: str,
        dst: str,
        edge_type: EdgeType,
        user_id: Optional[str] = None,
    ) -> None:
        self.backend.remove_edge(src, dst, edge_type)
        self._emit(
            "edge_pruned",
            f"{src}->{dst}",
            user_id,
            {"type": edge_type.value},
        )

    # ---- queries -------------------------------------------------------
    def neighbours(
        self,
        memory_id: str,
        *,
        depth: Optional[int] = None,
        edge_types: Optional[Iterable[EdgeType]] = None,
        min_weight: Optional[float] = None,
        include_archived: bool = False,
    ) -> List[MemoryEdge]:
        resolved_depth = depth if depth is not None else self.config.retrieval.expansion_depth
        resolved_min_weight = (
            min_weight
            if min_weight is not None
            else self.config.retrieval.edge_weight_floor
        )
        return self.backend.neighbours(
            memory_id,
            depth=resolved_depth,
            edge_types=edge_types,
            min_weight=resolved_min_weight,
            include_archived=include_archived,
        )

    def iter_embeddings(self) -> Iterable[Tuple[str, List[float]]]:
        return self.backend.iter_embeddings()

    def get_embedding(self, memory_id: str) -> Optional[List[float]]:
        return self.backend.get_embedding(memory_id)

    # ---- decay / housekeeping -----------------------------------------
    def tick_decay(self, now: Optional[datetime] = None) -> Dict[str, int]:
        """Apply exponential decay once; prune edges/archive stale nodes."""
        if not self.config.decay.enabled:
            return {"decayed": 0, "pruned": 0, "archived": 0}
        now = now or _utcnow()
        half_life_hours = self.config.decay.half_life_days * 24.0
        if half_life_hours <= 0:
            return {"decayed": 0, "pruned": 0, "archived": 0}
        lam = math.log(2.0) / half_life_hours
        min_edge_weight = self.config.decay.min_edge_weight
        min_node_relevance = self.config.decay.min_node_relevance

        decayed = 0
        pruned = 0
        archived = 0
        with self._lock:
            sticky_edges = {EdgeType.CAUSAL, EdgeType.CONTRADICTS}
            for edge in list(self.backend.all_edges()):
                if edge.type in sticky_edges:
                    continue  # causal + CONTRADICTS  are sticky
                delta_hours = max(0.0, (now - edge.updated_at).total_seconds() / 3600.0)
                if delta_hours == 0:
                    continue
                new_weight = edge.weight * math.exp(-lam * delta_hours)
                if new_weight < min_edge_weight:
                    self.backend.remove_edge(edge.src, edge.dst, edge.type)
                    pruned += 1
                    self._emit(
                        "edge_pruned",
                        f"{edge.src}->{edge.dst}",
                        None,
                        {"type": edge.type.value, "weight": new_weight},
                    )
                elif abs(new_weight - edge.weight) > 1e-6:
                    self.backend.upsert_edge(
                        MemoryEdge(
                            src=edge.src,
                            dst=edge.dst,
                            type=edge.type,
                            weight=new_weight,
                            created_at=edge.created_at,
                            updated_at=now,
                            evidence=edge.evidence,
                            metadata=edge.metadata,
                        )
                    )
                    decayed += 1
                    self._emit(
                        "edge_decayed",
                        f"{edge.src}->{edge.dst}",
                        None,
                        {"type": edge.type.value, "weight": new_weight},
                    )
            # Node relevance = 0.8 * max(live edge weight) + 0.2 * recency.
            # Nodes with no live edges AND relevance below floor get archived
            # (soft delete only; the vector-store row is preserved).
            if min_node_relevance > 0:
                try:
                    nodes = list(self.backend.iter_nodes(include_archived=False))
                except Exception as exc:  # pragma: no cover - backend dependent
                    logger.debug("iter_nodes() failed during tick_decay: %s", exc)
                    nodes = []
                for node in nodes:
                    if node.archived:
                        continue
                    try:
                        incident = list(self.backend.edges_for_node(node.id))
                    except Exception:  # pragma: no cover
                        incident = []
                    live_weights = [
                        e.weight for e in incident if e.weight >= min_edge_weight
                    ]
                    max_edge = max(live_weights) if live_weights else 0.0
                    anchor = node.last_accessed_at or node.updated_at
                    recency_hours = max(0.0, (now - anchor).total_seconds() / 3600.0)
                    recency = math.exp(-lam * recency_hours)
                    relevance = 0.8 * max_edge + 0.2 * recency
                    floor = float(getattr(node, "relevance_floor", 0.0) or 0.0)
                    effective = max(floor, relevance)
                    node.relevance = max(0.0, min(1.0, effective))
                    self.backend.upsert_node(node)
                    if (
                        not live_weights
                        and effective < min_node_relevance
                        and floor < min_node_relevance
                    ):
                        self.backend.archive_node(node.id)
                        archived += 1
                        self._emit(
                            "node_archived",
                            node.id,
                            node.user_id,
                            {"relevance": effective},
                        )
            self._last_decay_tick = now
        return {"decayed": decayed, "pruned": pruned, "archived": archived}

    def maybe_tick_decay(self) -> None:
        """Cheap decay trigger safe to call on every read."""
        if not self.config.decay.enabled or not self.config.decay.tick_on_read:
            return
        now = _utcnow()
        if self._last_decay_tick is None:
            self.tick_decay(now)
            return
        # Only run once per hour to keep read-path overhead bounded.
        if (now - self._last_decay_tick).total_seconds() >= 3600:
            self.tick_decay(now)

    # ---- persistence ---------------------------------------------------
    def snapshot(self, path: Optional[str] = None) -> None:
        target = path or self.config.persist_path
        if not target:
            return
        self.backend.snapshot(target)

    def export_jsonl(self, path: str) -> str:
        """Write a portable JSON-Lines snapshot of the current graph."""
        return self.backend.export_jsonl(path)

    def import_jsonl(self, path: str) -> Dict[str, int]:
        """Load a JSON-Lines snapshot produced by :meth:`export_jsonl`."""
        return self.backend.import_jsonl(path)

    def stats(self) -> Dict[str, int]:
        return {
            "nodes": self.backend.node_count(),
            "nodes_total": self.backend.node_count(include_archived=True),
            "edges": self.backend.edge_count(),
        }

    # ---- internal ------------------------------------------------------
    def _emit(
        self,
        event_type: str,
        target_id: str,
        user_id: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        if not self.config.log_changes or self.changelog is None:
            return
        try:
            self.changelog.append(
                ChangeEvent(
                    event_type=event_type,
                    target_id=target_id,
                    timestamp=_utcnow(),
                    user_id=user_id,
                    payload=payload,
                )
            )
        except Exception as exc:  # pragma: no cover - log-only
            logger.debug("ChangeLog append failed: %s", exc)
