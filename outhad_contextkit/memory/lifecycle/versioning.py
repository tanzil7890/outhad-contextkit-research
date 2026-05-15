"""Phase D4 — opt-in immutable versioning.

When ``decay_v2.versioning.mode='immutable'``, every ``Memory.update``
becomes an append: a new vector-store row is written with a fresh id;
the previous row's metadata gains ``status='superseded'``; the CGL
``UPDATED_FROM`` edge is reused (it already exists for overwrite
mode).

The version chain is stored in two places:
* On each CGL node: ``version`` int + ``prev_version_id`` link.
* As ``VersionRecord`` objects derived from the chain at read time
  (``VersionedMemoryStore.list_versions``). The store itself does not
  add a third storage table — the CGL node graph is the source of
  truth so we don't need a fourth SQLite store on top.

The default (``mode='overwrite'``) preserves today's contract: the
update mutates the row in place; the CGL builder still synthesises an
``UPDATED_FROM`` edge whenever the hash changes.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from outhad_contextkit.memory.context_graph.types import EdgeType, MemoryNode

logger = logging.getLogger(__name__)


@dataclass
class VersionRecord:
    """One version in a memory's history chain."""

    memory_id: str
    version: int
    prev_id: Optional[str]
    hash: str
    created_at: datetime
    superseded: bool = False
    actor_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VersionedMemoryStore:
    """Immutable-mode helper. Stateless — every method goes through
    the live ``Memory`` instance so the chain stays in sync with the
    rest of the storage layers.
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    # ------------------------------------------------------------------
    #  Writes
    # ------------------------------------------------------------------
    def insert_version(
        self,
        prev_id: str,
        new_data: str,
        new_metadata: Dict[str, Any],
        *,
        embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> VersionRecord:
        """Append a new version. Returns the resulting :class:`VersionRecord`.

        The new memory_id is freshly generated; the old row stays in
        place but its metadata gains ``status='superseded'`` so search
        callers can hide it transparently. CGL UPDATED_FROM edges are
        emitted automatically by the builder hook so the version chain
        is queryable both via the CGL graph and via ``list_versions``.
        """
        memory = self._memory
        prev_node = memory._context_graph.backend.get_node(prev_id) if (
            memory._context_graph
        ) else None
        prev_hash = getattr(prev_node, "hash", "") if prev_node else ""
        new_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        new_hash = _hash_text(new_data)
        embed = embedding
        if embed is None and getattr(memory, "embedding_model", None) is not None:
            try:
                embed = memory.embedding_model.embed(new_data, "update")
            except Exception as exc:  # pragma: no cover - embed is best-effort
                logger.debug("immutable embed failed: %s", exc)
                embed = None

        # 1) Insert the new row.
        meta = dict(new_metadata or {})
        meta["data"] = new_data
        meta["hash"] = new_hash
        meta["created_at"] = now.isoformat()
        meta["updated_at"] = now.isoformat()
        meta.setdefault("status", "active")
        meta["prev_version_id"] = prev_id
        meta["version_root"] = meta.get("version_root") or prev_id
        try:
            memory.vector_store.insert(
                vectors=[embed] if embed is not None else None,
                ids=[new_id],
                payloads=[meta],
            )
        except Exception as exc:
            logger.error("VersionedMemoryStore vector insert failed: %s", exc)
            raise

        # 2) Mark the old row superseded.
        try:
            existing = memory.vector_store.get(vector_id=prev_id)
            old_payload = dict(getattr(existing, "payload", {}) or {})
            old_payload["status"] = "superseded"
            old_payload["next_version_id"] = new_id
            memory.vector_store.update(
                vector_id=prev_id,
                payload=old_payload,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Mark previous version superseded failed for %s: %s",
                prev_id,
                exc,
            )

        # 3) History row + CGL upsert with prev_version link.
        try:
            memory.db.add_history(
                new_id, None, new_data, "ADD",
                created_at=meta["created_at"],
                updated_at=meta["updated_at"],
                actor_id=actor_id,
                role=None,
                tenant_id=meta.get("tenant_id"),
                sub_tenant_id=meta.get("sub_tenant_id"),
            )
            memory.db.add_history(
                prev_id,
                _payload_text(prev_node, meta_default=new_data),
                new_data,
                "UPDATE",
                created_at=meta["created_at"],
                updated_at=meta["updated_at"],
                actor_id=actor_id,
                role=None,
                tenant_id=meta.get("tenant_id"),
                sub_tenant_id=meta.get("sub_tenant_id"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("VersionedMemoryStore history write failed: %s", exc)

        if memory._context_graph is not None:
            cg = memory._context_graph
            try:
                node = cg.upsert_memory_node(
                    new_id,
                    new_data,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    metadata=meta,
                    embedding=embed,
                    tenant_id=meta.get("tenant_id"),
                    sub_tenant_id=meta.get("sub_tenant_id"),
                )
                # Stamp the version chain explicitly so the new node
                # carries (version=prev_version+1, prev_version_id=prev).
                if prev_node is not None:
                    node.version = int(getattr(prev_node, "version", 1) or 1) + 1
                    node.prev_version_id = prev_id
                    cg.backend.upsert_node(node)
                cg.upsert_edge(
                    prev_id,
                    new_id,
                    EdgeType.UPDATED_FROM,
                    weight=1.0,
                    user_id=user_id,
                    metadata={"reason": "immutable_update"},
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("CGL versioning upsert failed: %s", exc)

        return VersionRecord(
            memory_id=new_id,
            version=int(getattr(prev_node, "version", 1) or 1) + 1,
            prev_id=prev_id,
            hash=new_hash,
            created_at=now,
            superseded=False,
            actor_id=actor_id,
            metadata=dict(meta),
        )

    # ------------------------------------------------------------------
    #  Reads
    # ------------------------------------------------------------------
    def list_versions(self, memory_id: str) -> List[VersionRecord]:
        """Return the ``[oldest, ..., newest]`` chain that contains ``memory_id``."""
        memory = self._memory
        if memory._context_graph is None:
            return []

        # Walk backwards via prev_version_id to find the root.
        backend = memory._context_graph.backend
        node = backend.get_node(memory_id)
        if node is None:
            return []
        chain: List[MemoryNode] = []
        seen = set()
        cursor: Optional[MemoryNode] = node
        while cursor is not None and cursor.id not in seen:
            chain.append(cursor)
            seen.add(cursor.id)
            prev = getattr(cursor, "prev_version_id", None)
            if not prev:
                break
            cursor = backend.get_node(prev)
        chain.reverse()  # oldest → newest

        # Walk forward from the last node via vector-store metadata
        # (next_version_id is set on supersede). This catches versions
        # created after `memory_id`.
        forward_cursor = chain[-1].id
        for _ in range(64):  # bound the walk for safety
            try:
                row = memory.vector_store.get(vector_id=forward_cursor)
            except Exception:
                break
            if row is None:
                break
            payload = getattr(row, "payload", {}) or {}
            nxt = payload.get("next_version_id")
            if not nxt or nxt in seen:
                break
            nxt_node = backend.get_node(nxt)
            if nxt_node is None:
                break
            chain.append(nxt_node)
            seen.add(nxt)
            forward_cursor = nxt

        records: List[VersionRecord] = []
        for n in chain:
            try:
                vrow = memory.vector_store.get(vector_id=n.id)
                superseded = bool(
                    (getattr(vrow, "payload", {}) or {}).get("status")
                    == "superseded"
                )
            except Exception:
                superseded = False
            records.append(
                VersionRecord(
                    memory_id=n.id,
                    version=int(getattr(n, "version", 1) or 1),
                    prev_id=getattr(n, "prev_version_id", None),
                    hash=getattr(n, "hash", ""),
                    created_at=getattr(n, "created_at", datetime.now(timezone.utc)),
                    superseded=superseded,
                    metadata=dict(getattr(n, "metadata", {}) or {}),
                )
            )
        return records

    def latest(self, memory_id: str) -> Optional[VersionRecord]:
        chain = self.list_versions(memory_id)
        if not chain:
            return None
        # Latest = newest non-superseded; fallback to last in chain.
        for v in reversed(chain):
            if not v.superseded:
                return v
        return chain[-1]

    def prune(self, memory_id: str, *, keep: int) -> int:
        """Drop chain elements older than ``keep`` (FIFO).

        The oldest superseded versions are removed first; their CGL
        nodes are deleted and vector-store rows dropped. Active
        (non-superseded) versions are NEVER pruned.
        """
        chain = self.list_versions(memory_id)
        if len(chain) <= int(max(1, keep)):
            return 0
        memory = self._memory
        excess = len(chain) - int(max(1, keep))
        removed = 0
        for record in chain[:excess]:
            if not record.superseded:
                continue
            try:
                memory.vector_store.delete(vector_id=record.memory_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "prune vector delete failed for %s: %s", record.memory_id, exc
                )
            if memory._context_graph is not None:
                try:
                    memory._context_graph.delete_memory_node(record.memory_id)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "prune CGL delete failed for %s: %s",
                        record.memory_id,
                        exc,
                    )
            removed += 1
        return removed


def _hash_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def _payload_text(node: Optional[MemoryNode], *, meta_default: str) -> str:
    """Best-effort previous-text recovery for the history row."""
    if node is None:
        return meta_default
    meta = getattr(node, "metadata", {}) or {}
    return str(meta.get("data") or meta_default)


__all__ = ["VersionedMemoryStore", "VersionRecord"]
