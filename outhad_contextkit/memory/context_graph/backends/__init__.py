"""Context-Graph backend abstract base class."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from outhad_contextkit.memory.context_graph.types import EdgeType, MemoryEdge, MemoryNode


class ContextGraphBackend(ABC):
    """Storage-agnostic interface for the Context-Graph Layer.

    Two concrete implementations ship with the library:

    * :class:`NetworkXBackend` — in-process, zero extra infra (default).
    * :class:`Neo4jBackend` — reuses the existing ``MemoryGraph`` driver
      when the user already enabled Neo4j for the entity graph.
    """

    # ---- nodes ---------------------------------------------------------
    @abstractmethod
    def upsert_node(self, node: MemoryNode) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[MemoryNode]: ...

    @abstractmethod
    def archive_node(self, node_id: str) -> None: ...

    @abstractmethod
    def delete_node(self, node_id: str) -> None: ...

    # ---- edges ---------------------------------------------------------
    @abstractmethod
    def upsert_edge(self, edge: MemoryEdge) -> None: ...

    @abstractmethod
    def remove_edge(self, src: str, dst: str, type: EdgeType) -> None: ...

    @abstractmethod
    def neighbours(
        self,
        node_id: str,
        depth: int = 1,
        edge_types: Optional[Iterable[EdgeType]] = None,
        min_weight: float = 0.0,
        include_archived: bool = False,
    ) -> List[MemoryEdge]: ...

    @abstractmethod
    def all_edges(self) -> Iterable[MemoryEdge]: ...

    def iter_nodes(self, include_archived: bool = True) -> Iterable[MemoryNode]:
        """Yield every node; default impl returns empty so optional backends
        can skip implementing it. Required for node-level decay / archival."""
        return iter(())

    def edges_for_node(self, node_id: str) -> Iterable[MemoryEdge]:
        """Return every edge incident to ``node_id`` (in either direction)."""
        return iter(())

    # ---- frequency tracking  --------------------------------
    def bump_access_count(
        self,
        node_id: str,
        *,
        delta: int = 1,
        now: Optional[datetime] = None,
    ) -> Optional[int]:
        """Increment ``node.access_count`` by ``delta`` and refresh
        ``last_accessed_at``. Returns the new count, or ``None`` when the
        node is unknown.

        Default implementation performs read → mutate → upsert. Backends
        should override with an atomic update when possible (e.g. Neo4j
        does a single Cypher ``SET``).
        """
        node = self.get_node(node_id)
        if node is None:
            return None
        new_count = int(getattr(node, "access_count", 0) or 0) + int(delta)
        if new_count < 0:
            new_count = 0
        node.access_count = new_count
        node.last_accessed_at = now or datetime.utcnow()
        self.upsert_node(node)
        return new_count

    def max_access_count(self) -> int:
        """Return the max ``access_count`` over all live nodes.

        Default implementation iterates :meth:`iter_nodes`. Backends that
        can run a single aggregation query (Neo4j ``MAX()``) should override.
        """
        peak = 0
        try:
            for node in self.iter_nodes(include_archived=False):
                count = int(getattr(node, "access_count", 0) or 0)
                if count > peak:
                    peak = count
        except Exception:  # pragma: no cover - defensive
            return peak
        return peak

    # ---- importance scoring (extension) -----------------------
    def bump_relevance(
        self,
        node_id: str,
        delta: float,
        *,
        floor: Optional[float] = None,
        cap: float = 1.0,
    ) -> Optional[float]:
        """Adjust a node's ``relevance`` by ``delta`` and optionally raise its
        ``relevance_floor``.

        Returns the new relevance, or ``None`` if the node is unknown.
        Default implementation performs read → clamp → upsert and is safe for
        any backend that implements :meth:`get_node` / :meth:`upsert_node`.
        Backends may override for atomic updates.
        """
        node = self.get_node(node_id)
        if node is None:
            return None
        new_rel = max(0.0, min(float(cap), float(node.relevance) + float(delta)))
        node.relevance = new_rel
        if floor is not None:
            node.relevance_floor = max(float(node.relevance_floor), float(floor))
        self.upsert_node(node)
        return new_rel

    # ---- embedding sidecar (optional; topic-similarity uses this) -------
    def set_embedding(self, node_id: str, vector: List[float]) -> None:  # noqa: D401
        """Default no-op; override if the backend stores embeddings."""
        return None

    def get_embedding(self, node_id: str) -> Optional[List[float]]:
        return None

    def iter_embeddings(self) -> Iterable[tuple]:
        """Yield ``(node_id, vector)`` pairs for topic-similarity search."""
        return iter(())

    # ---- persistence ---------------------------------------------------
    @abstractmethod
    def snapshot(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    # ---- portable JSON-Lines export / import  -----------------
    #
    # These are non-abstract, backend-agnostic helpers built on
    # :meth:`iter_nodes` / :meth:`all_edges` / :meth:`iter_embeddings`.
    # Concrete backends may override for a streaming implementation, but
    # the defaults below are correct for any backend that keeps those
    # four methods in sync.
    _EXPORT_ENVELOPE_VERSION = 1

    def export_jsonl(self, path: str) -> str:
        """Write a portable JSON-Lines snapshot. Returns the path written.

        Format (v1):
            line 0: envelope metadata
            line 1..N: ``{"kind": "node" | "edge" | "embedding", "data": ...}``
        """
        nodes = list(self.iter_nodes(include_archived=True))
        edges = list(self.all_edges())
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{path}.tmp"
        envelope = {
            "__envelope__": self._EXPORT_ENVELOPE_VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "backend": self.__class__.__name__,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(envelope, sort_keys=True) + "\n")
            for node in sorted(nodes, key=lambda n: n.id):
                fh.write(
                    json.dumps(
                        {"kind": "node", "data": _node_to_dict(node)},
                        sort_keys=True,
                    )
                    + "\n"
                )
            for edge in sorted(
                edges, key=lambda e: (e.src, e.dst, e.type.value)
            ):
                fh.write(
                    json.dumps(
                        {"kind": "edge", "data": _edge_to_dict(edge)},
                        sort_keys=True,
                    )
                    + "\n"
                )
            for node_id, vector in sorted(self.iter_embeddings()):
                fh.write(
                    json.dumps(
                        {
                            "kind": "embedding",
                            "data": {
                                "id": node_id,
                                "vector": list(vector),
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(tmp_path, path)
        return path

    def import_jsonl(self, path: str) -> Dict[str, int]:
        """Load a JSON-Lines snapshot written by :meth:`export_jsonl`.

        Safe against unknown ``kind`` values (forward compatible).
        Returns a dict with counts of applied entries.
        """
        counts = {"nodes": 0, "edges": 0, "embeddings": 0, "skipped": 0}
        if not os.path.exists(path):
            return counts
        with open(path, "r", encoding="utf-8") as fh:
            header = fh.readline()
            try:
                envelope = json.loads(header) if header else {}
            except json.JSONDecodeError:
                envelope = {}
            if "__envelope__" not in envelope:
                counts["skipped"] += 1
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    counts["skipped"] += 1
                    continue
                kind = entry.get("kind")
                data = entry.get("data") or {}
                if kind == "node":
                    self.upsert_node(_dict_to_node(data))
                    counts["nodes"] += 1
                elif kind == "edge":
                    self.upsert_edge(_dict_to_edge(data))
                    counts["edges"] += 1
                elif kind == "embedding":
                    vec = data.get("vector") or []
                    nid = data.get("id")
                    if nid:
                        self.set_embedding(nid, list(vec))
                        counts["embeddings"] += 1
                    else:
                        counts["skipped"] += 1
                else:
                    counts["skipped"] += 1
        return counts

    # ---- stats ---------------------------------------------------------
    @abstractmethod
    def node_count(self, include_archived: bool = False) -> int: ...

    @abstractmethod
    def edge_count(self) -> int: ...


# ----------------------------------------------------------------------
# Serialisation helpers (module-level so backends and ``replay.py`` can
# share a single JSON <-> dataclass contract).
# ----------------------------------------------------------------------


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _parse_iso(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _node_to_dict(node: MemoryNode) -> Dict[str, Any]:
    return {
        "id": node.id,
        "hash": node.hash,
        "created_at": _iso(node.created_at),
        "updated_at": _iso(node.updated_at),
        "version": int(node.version),
        "prev_version_id": node.prev_version_id,
        "relevance": float(node.relevance),
        "relevance_floor": float(node.relevance_floor),
        "archived": bool(node.archived),
        "user_id": node.user_id,
        "agent_id": node.agent_id,
        "run_id": node.run_id,
        "last_accessed_at": _iso(node.last_accessed_at),
        "access_count": int(getattr(node, "access_count", 0) or 0),
        "helpful_count": int(getattr(node, "helpful_count", 0) or 0),
        "unhelpful_count": int(getattr(node, "unhelpful_count", 0) or 0),
        "last_feedback_at": _iso(getattr(node, "last_feedback_at", None)),
        "tenant_id": getattr(node, "tenant_id", None),
        "sub_tenant_id": getattr(node, "sub_tenant_id", None),
        "metadata": dict(node.metadata or {}),
    }


def _dict_to_node(data: Dict[str, Any]) -> MemoryNode:
    now = datetime.utcnow()
    return MemoryNode(
        id=str(data["id"]),
        hash=str(data.get("hash", "")),
        created_at=_parse_iso(data.get("created_at")) or now,
        updated_at=_parse_iso(data.get("updated_at")) or now,
        version=int(data.get("version", 1)),
        prev_version_id=data.get("prev_version_id"),
        relevance=float(data.get("relevance", 1.0)),
        relevance_floor=float(data.get("relevance_floor", 0.0)),
        archived=bool(data.get("archived", False)),
        user_id=data.get("user_id"),
        agent_id=data.get("agent_id"),
        run_id=data.get("run_id"),
        last_accessed_at=_parse_iso(data.get("last_accessed_at")),
        access_count=int(data.get("access_count", 0) or 0),
        helpful_count=int(data.get("helpful_count", 0) or 0),
        unhelpful_count=int(data.get("unhelpful_count", 0) or 0),
        last_feedback_at=_parse_iso(data.get("last_feedback_at")),
        tenant_id=data.get("tenant_id"),
        sub_tenant_id=data.get("sub_tenant_id"),
        metadata=dict(data.get("metadata") or {}),
    )


def _edge_to_dict(edge: MemoryEdge) -> Dict[str, Any]:
    return {
        "src": edge.src,
        "dst": edge.dst,
        "type": edge.type.value,
        "weight": float(edge.weight),
        "created_at": _iso(edge.created_at),
        "updated_at": _iso(edge.updated_at),
        "evidence": edge.evidence,
        "metadata": dict(edge.metadata or {}),
    }


def _dict_to_edge(data: Dict[str, Any]) -> MemoryEdge:
    now = datetime.utcnow()
    return MemoryEdge(
        src=str(data["src"]),
        dst=str(data["dst"]),
        type=EdgeType(data.get("type", EdgeType.REPLY_TO.value)),
        weight=float(data.get("weight", 1.0)),
        created_at=_parse_iso(data.get("created_at")) or now,
        updated_at=_parse_iso(data.get("updated_at")) or now,
        evidence=data.get("evidence"),
        metadata=dict(data.get("metadata") or {}),
    )
