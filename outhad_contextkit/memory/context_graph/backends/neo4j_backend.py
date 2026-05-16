"""Neo4j-backed implementation of :class:`ContextGraphBackend`.

This backend is only used when the user has already opted into Neo4j via the
existing entity-graph configuration. It reuses the driver exposed by
:class:`~outhad_contextkit.memory.graph_memory.MemoryGraph` so no extra network
connection is created.

Schema:

* Nodes carry the label ``:MemoryNode`` and are keyed on the memory UUID.
* Edges use the relationship type ``CTX_EDGE`` with a ``type`` property that
  mirrors :class:`EdgeType`.

Snapshots are implemented as a JSON export of the typed projection so the
backend can be rebuilt on a different Neo4j instance. For very large graphs
operators should rely on native ``neo4j-admin`` dumps instead.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from outhad_contextkit.memory.context_graph.backends import ContextGraphBackend
from outhad_contextkit.memory.context_graph.types import EdgeType, MemoryEdge, MemoryNode

logger = logging.getLogger(__name__)

_LABEL = "MemoryNode"
_REL = "CTX_EDGE"


def _dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


class Neo4jBackend(ContextGraphBackend):
    """Neo4j-backed context graph that piggy-backs on the existing driver."""

    def __init__(self, driver: Any) -> None:
        """Accepts either a bare Neo4j driver or the :class:`MemoryGraph`.

        The entity graph's ``MemoryGraph`` exposes a ``.graph`` attribute which
        wraps the driver; both shapes are accepted so callers don't have to
        worry about unwrapping.
        """
        self._driver = driver
        self._ensure_schema()

    # ---- internal helpers ----------------------------------------------
    def _execute(self, query: str, params: Optional[dict] = None) -> List[Any]:
        params = params or {}
        # Support langchain's Neo4jGraph wrapper (has .query) and a raw driver.
        if hasattr(self._driver, "query"):
            try:
                return self._driver.query(query, params=params) or []
            except TypeError:
                return self._driver.query(query, params) or []
        with self._driver.session() as session:  # pragma: no cover - live Neo4j
            result = session.run(query, **params)
            return [record.data() for record in result]

    def _ensure_schema(self) -> None:
        try:
            self._execute(
                f"CREATE CONSTRAINT context_memory_node_id IF NOT EXISTS "
                f"FOR (n:{_LABEL}) REQUIRE n.id IS UNIQUE"
            )
            self._execute(
                f"CREATE INDEX context_memory_node_archived IF NOT EXISTS "
                f"FOR (n:{_LABEL}) ON (n.archived)"
            )
        except Exception as exc:  # pragma: no cover - depends on Neo4j version
            logger.debug("Neo4jBackend schema bootstrap skipped: %s", exc)

    @staticmethod
    def _node_to_props(node: MemoryNode) -> Dict[str, Any]:
        return {
            "id": node.id,
            "hash": node.hash,
            "created_at": node.created_at.isoformat() if node.created_at else None,
            "updated_at": node.updated_at.isoformat() if node.updated_at else None,
            "version": node.version,
            "prev_version_id": node.prev_version_id,
            "relevance": float(node.relevance),
            "relevance_floor": float(node.relevance_floor),
            "archived": bool(node.archived),
            "user_id": node.user_id,
            "agent_id": node.agent_id,
            "run_id": node.run_id,
            "last_accessed_at": node.last_accessed_at.isoformat()
            if node.last_accessed_at
            else None,
            "access_count": int(getattr(node, "access_count", 0) or 0),
            "helpful_count": int(getattr(node, "helpful_count", 0) or 0),
            "unhelpful_count": int(getattr(node, "unhelpful_count", 0) or 0),
            "last_feedback_at": node.last_feedback_at.isoformat()
            if getattr(node, "last_feedback_at", None)
            else None,
            "tenant_id": getattr(node, "tenant_id", None),
            "sub_tenant_id": getattr(node, "sub_tenant_id", None),
            "metadata_json": json.dumps(node.metadata or {}),
        }

    @staticmethod
    def _props_to_node(props: Dict[str, Any]) -> MemoryNode:
        return MemoryNode(
            id=props["id"],
            hash=props.get("hash", ""),
            created_at=_dt(props.get("created_at")) or datetime.utcnow(),
            updated_at=_dt(props.get("updated_at")) or datetime.utcnow(),
            version=int(props.get("version", 1)),
            prev_version_id=props.get("prev_version_id"),
            relevance=float(props.get("relevance", 1.0)),
            relevance_floor=float(props.get("relevance_floor", 0.0)),
            archived=bool(props.get("archived", False)),
            user_id=props.get("user_id"),
            agent_id=props.get("agent_id"),
            run_id=props.get("run_id"),
            last_accessed_at=_dt(props.get("last_accessed_at")),
            access_count=int(props.get("access_count", 0) or 0),
            helpful_count=int(props.get("helpful_count", 0) or 0),
            unhelpful_count=int(props.get("unhelpful_count", 0) or 0),
            last_feedback_at=_dt(props.get("last_feedback_at")),
            tenant_id=props.get("tenant_id"),
            sub_tenant_id=props.get("sub_tenant_id"),
            metadata=json.loads(props.get("metadata_json") or "{}"),
        )

    @staticmethod
    def _edge_to_props(edge: MemoryEdge) -> Dict[str, Any]:
        return {
            "type": edge.type.value,
            "weight": float(edge.weight),
            "created_at": edge.created_at.isoformat() if edge.created_at else None,
            "updated_at": edge.updated_at.isoformat() if edge.updated_at else None,
            "evidence": edge.evidence,
            "metadata_json": json.dumps(edge.metadata or {}),
        }

    @staticmethod
    def _props_to_edge(src: str, dst: str, props: Dict[str, Any]) -> MemoryEdge:
        return MemoryEdge(
            src=src,
            dst=dst,
            type=EdgeType(props.get("type", EdgeType.REPLY_TO.value)),
            weight=float(props.get("weight", 1.0)),
            created_at=_dt(props.get("created_at")) or datetime.utcnow(),
            updated_at=_dt(props.get("updated_at")) or datetime.utcnow(),
            evidence=props.get("evidence"),
            metadata=json.loads(props.get("metadata_json") or "{}"),
        )

    # ---- nodes ---------------------------------------------------------
    def upsert_node(self, node: MemoryNode) -> None:
        query = (
            f"MERGE (n:{_LABEL} {{id: $id}}) "
            "SET n += $props"
        )
        self._execute(query, {"id": node.id, "props": self._node_to_props(node)})

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        query = f"MATCH (n:{_LABEL} {{id: $id}}) RETURN properties(n) AS props"
        rows = self._execute(query, {"id": node_id})
        if not rows:
            return None
        props = rows[0].get("props") if isinstance(rows[0], dict) else None
        if not props:
            return None
        return self._props_to_node(props)

    def archive_node(self, node_id: str) -> None:
        query = f"MATCH (n:{_LABEL} {{id: $id}}) SET n.archived = true"
        self._execute(query, {"id": node_id})

    def delete_node(self, node_id: str) -> None:
        query = f"MATCH (n:{_LABEL} {{id: $id}}) DETACH DELETE n"
        self._execute(query, {"id": node_id})

    # ---- edges ---------------------------------------------------------
    def upsert_edge(self, edge: MemoryEdge) -> None:
        query = (
            f"MATCH (a:{_LABEL} {{id: $src}}), (b:{_LABEL} {{id: $dst}}) "
            f"MERGE (a)-[r:{_REL} {{type: $type}}]->(b) "
            "SET r += $props"
        )
        self._execute(
            query,
            {
                "src": edge.src,
                "dst": edge.dst,
                "type": edge.type.value,
                "props": self._edge_to_props(edge),
            },
        )

    def remove_edge(self, src: str, dst: str, type: EdgeType) -> None:
        query = (
            f"MATCH (a:{_LABEL} {{id: $src}})-[r:{_REL} {{type: $type}}]->(b:{_LABEL} {{id: $dst}}) "
            "DELETE r"
        )
        self._execute(query, {"src": src, "dst": dst, "type": type.value})

    def neighbours(
        self,
        node_id: str,
        depth: int = 1,
        edge_types: Optional[Iterable[EdgeType]] = None,
        min_weight: float = 0.0,
        include_archived: bool = False,
    ) -> List[MemoryEdge]:
        if depth < 1:
            return []
        types_clause = ""
        params: Dict[str, Any] = {
            "id": node_id,
            "min_weight": float(min_weight),
        }
        if edge_types is not None:
            params["types"] = [
                et.value if isinstance(et, EdgeType) else EdgeType(et).value
                for et in edge_types
            ]
            types_clause = "AND r.type IN $types "
        archived_clause = "" if include_archived else "AND coalesce(b.archived, false) = false "
        query = (
            f"MATCH (a:{_LABEL} {{id: $id}})-[r:{_REL}*1..{int(depth)}]-(b:{_LABEL}) "
            "WITH relationships(r) AS rels, b "
            "UNWIND rels AS rel "
            "WITH rel, startNode(rel) AS s, endNode(rel) AS e "
            "WHERE rel.weight >= $min_weight "
            f"{types_clause}"
            f"{archived_clause.replace('b.archived', 'e.archived')}"
            "RETURN DISTINCT s.id AS src, e.id AS dst, properties(rel) AS props"
        )
        rows = self._execute(query, params)
        edges: List[MemoryEdge] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            edges.append(self._props_to_edge(row["src"], row["dst"], row.get("props") or {}))
        return edges

    def all_edges(self) -> Iterable[MemoryEdge]:
        query = (
            f"MATCH (a:{_LABEL})-[r:{_REL}]->(b:{_LABEL}) "
            "RETURN a.id AS src, b.id AS dst, properties(r) AS props"
        )
        rows = self._execute(query)
        return [
            self._props_to_edge(row["src"], row["dst"], row.get("props") or {})
            for row in rows
            if isinstance(row, dict)
        ]

    # ---- persistence ---------------------------------------------------
    def snapshot(self, path: str) -> None:
        nodes_q = f"MATCH (n:{_LABEL}) RETURN properties(n) AS props"
        edges_q = (
            f"MATCH (a:{_LABEL})-[r:{_REL}]->(b:{_LABEL}) "
            "RETURN a.id AS src, b.id AS dst, properties(r) AS props"
        )
        nodes = [row.get("props") for row in self._execute(nodes_q) if isinstance(row, dict)]
        edges = [
            {"src": row["src"], "dst": row["dst"], "props": row.get("props") or {}}
            for row in self._execute(edges_q)
            if isinstance(row, dict)
        ]
        payload = {"version": 1, "nodes": nodes, "edges": edges}
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_path, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for props in payload.get("nodes", []):
            if not props:
                continue
            self.upsert_node(self._props_to_node(props))
        for entry in payload.get("edges", []):
            edge = self._props_to_edge(entry["src"], entry["dst"], entry.get("props") or {})
            self.upsert_edge(edge)

    # ---- iter_nodes ----------------------------------------------------
    def iter_nodes(self, include_archived: bool = True) -> Iterable[MemoryNode]:
        if include_archived:
            query = f"MATCH (n:{_LABEL}) RETURN properties(n) AS props"
            params: Dict[str, Any] = {}
        else:
            query = (
                f"MATCH (n:{_LABEL}) "
                "WHERE coalesce(n.archived, false) = false "
                "RETURN properties(n) AS props"
            )
            params = {}
        rows = self._execute(query, params)
        nodes: List[MemoryNode] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            props = row.get("props")
            if props:
                nodes.append(self._props_to_node(props))
        return nodes

    # ---- edges_for_node ------------------------------------------------
    def edges_for_node(self, node_id: str) -> Iterable[MemoryEdge]:
        query = (
            f"MATCH (a:{_LABEL})-[r:{_REL}]-(b:{_LABEL}) "
            "WHERE a.id = $id OR b.id = $id "
            "RETURN startNode(r).id AS src, endNode(r).id AS dst, properties(r) AS props"
        )
        rows = self._execute(query, {"id": node_id})
        edges: List[MemoryEdge] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            edges.append(
                self._props_to_edge(row["src"], row["dst"], row.get("props") or {})
            )
        return edges

    # ---- embedding sidecar ---------------------------------------------
    def set_embedding(self, node_id: str, vector: List[float]) -> None:
        query = (
            f"MATCH (n:{_LABEL} {{id: $id}}) "
            "SET n.embedding_json = $vec"
        )
        self._execute(query, {"id": node_id, "vec": json.dumps(vector)})

    def get_embedding(self, node_id: str) -> Optional[List[float]]:
        query = (
            f"MATCH (n:{_LABEL} {{id: $id}}) "
            "RETURN n.embedding_json AS vec"
        )
        rows = self._execute(query, {"id": node_id})
        if not rows:
            return None
        row = rows[0]
        raw = row.get("vec") if isinstance(row, dict) else None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def iter_embeddings(self) -> Iterable[Tuple[str, List[float]]]:
        query = (
            f"MATCH (n:{_LABEL}) "
            "WHERE n.embedding_json IS NOT NULL "
            "RETURN n.id AS id, n.embedding_json AS vec"
        )
        rows = self._execute(query)
        result: List[Tuple[str, List[float]]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw = row.get("vec")
            if not raw:
                continue
            try:
                result.append((row["id"], json.loads(raw)))
            except (TypeError, ValueError, KeyError):
                continue
        return result

    # ---- frequency tracking  --------------------------------
    def bump_access_count(
        self,
        node_id: str,
        *,
        delta: int = 1,
        now: Optional[datetime] = None,
    ) -> Optional[int]:
        ts = (now or datetime.utcnow()).isoformat()
        query = (
            f"MATCH (n:{_LABEL} {{id: $id}}) "
            "SET n.access_count = CASE "
            "  WHEN coalesce(n.access_count, 0) + $delta < 0 THEN 0 "
            "  ELSE coalesce(n.access_count, 0) + $delta END, "
            "n.last_accessed_at = $ts "
            "RETURN n.access_count AS cnt"
        )
        rows = self._execute(
            query,
            {"id": node_id, "delta": int(delta), "ts": ts},
        )
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, dict) and "cnt" in row:
            try:
                return int(row["cnt"])
            except (TypeError, ValueError):
                return None
        return None

    def max_access_count(self) -> int:
        query = (
            f"MATCH (n:{_LABEL}) "
            "WHERE coalesce(n.archived, false) = false "
            "RETURN coalesce(max(n.access_count), 0) AS peak"
        )
        rows = self._execute(query)
        if not rows:
            return 0
        row = rows[0]
        if isinstance(row, dict):
            try:
                return int(row.get("peak", 0) or 0)
            except (TypeError, ValueError):
                return 0
        return 0

    # ---- importance scoring  ---------------------------------
    def bump_relevance(
        self,
        node_id: str,
        delta: float,
        *,
        floor: Optional[float] = None,
        cap: float = 1.0,
    ) -> Optional[float]:
        query = (
            f"MATCH (n:{_LABEL} {{id: $id}}) "
            "SET n.relevance = CASE "
            "  WHEN coalesce(n.relevance, 1.0) + $delta > $cap THEN $cap "
            "  WHEN coalesce(n.relevance, 1.0) + $delta < 0.0 THEN 0.0 "
            "  ELSE coalesce(n.relevance, 1.0) + $delta END, "
            "n.relevance_floor = CASE "
            "  WHEN $floor IS NULL THEN coalesce(n.relevance_floor, 0.0) "
            "  WHEN $floor > coalesce(n.relevance_floor, 0.0) THEN $floor "
            "  ELSE coalesce(n.relevance_floor, 0.0) END "
            "RETURN n.relevance AS rel"
        )
        rows = self._execute(
            query,
            {
                "id": node_id,
                "delta": float(delta),
                "cap": float(cap),
                "floor": None if floor is None else float(floor),
            },
        )
        if not rows:
            return None
        row = rows[0]
        if isinstance(row, dict) and "rel" in row:
            try:
                return float(row["rel"])
            except (TypeError, ValueError):
                return None
        return None

    # ---- stats ---------------------------------------------------------
    def node_count(self, include_archived: bool = False) -> int:
        if include_archived:
            query = f"MATCH (n:{_LABEL}) RETURN count(n) AS c"
        else:
            query = (
                f"MATCH (n:{_LABEL}) WHERE coalesce(n.archived, false) = false "
                "RETURN count(n) AS c"
            )
        rows = self._execute(query)
        if not rows:
            return 0
        row = rows[0]
        if isinstance(row, dict):
            return int(row.get("c", 0))
        return 0

    def edge_count(self) -> int:
        rows = self._execute(f"MATCH ()-[r:{_REL}]->() RETURN count(r) AS c")
        if not rows:
            return 0
        row = rows[0]
        if isinstance(row, dict):
            return int(row.get("c", 0))
        return 0
