"""In-process ``networkx``-backed implementation of :class:`ContextGraphBackend`.

This is the default backend for the Context-Graph Layer. It requires no
external infrastructure and is safe to enable on any deployment that already
runs Outhad_ContextKit. Snapshot/load uses ``pickle`` which keeps serialisation
trivially compatible with the dataclasses defined in
:mod:`outhad_contextkit.memory.context_graph.types`.

Thread-safety: all mutating methods acquire a module-level ``RLock`` so the
backend is safe to share across the concurrent call-sites inside
:class:`~outhad_contextkit.memory.main.Memory` and :class:`AsyncMemory`.
"""
from __future__ import annotations

import logging
import os
import pickle
import threading
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - import guarded at facade
    raise ImportError(
        "The networkx backend requires the `networkx` package. Install it with "
        "`pip install networkx`."
    ) from exc

from outhad_contextkit.memory.context_graph.backends import ContextGraphBackend
from outhad_contextkit.memory.context_graph.types import EdgeType, MemoryEdge, MemoryNode

logger = logging.getLogger(__name__)


def _edge_key(edge_type: EdgeType) -> str:
    """MultiDiGraph keys must be hashable; use the enum value."""
    return edge_type.value


class NetworkXBackend(ContextGraphBackend):
    """Default in-process backend built on :class:`networkx.MultiDiGraph`."""

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._embeddings: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    # ---- nodes ---------------------------------------------------------
    def upsert_node(self, node: MemoryNode) -> None:
        with self._lock:
            self._graph.add_node(node.id, node=node)

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        with self._lock:
            data = self._graph.nodes.get(node_id)
            if not data:
                return None
            return data.get("node")

    def archive_node(self, node_id: str) -> None:
        with self._lock:
            data = self._graph.nodes.get(node_id)
            if not data:
                return
            node: Optional[MemoryNode] = data.get("node")
            if node is None:
                return
            node.archived = True
            self._graph.nodes[node_id]["node"] = node

    def delete_node(self, node_id: str) -> None:
        with self._lock:
            if self._graph.has_node(node_id):
                self._graph.remove_node(node_id)
            self._embeddings.pop(node_id, None)

    # ---- edges ---------------------------------------------------------
    def upsert_edge(self, edge: MemoryEdge) -> None:
        with self._lock:
            key = _edge_key(edge.type)
            if self._graph.has_edge(edge.src, edge.dst, key=key):
                existing = self._graph[edge.src][edge.dst][key].get("edge")
                if existing is not None:
                    existing.weight = edge.weight
                    existing.updated_at = edge.updated_at
                    if edge.evidence is not None:
                        existing.evidence = edge.evidence
                    if edge.metadata:
                        existing.metadata.update(edge.metadata)
                    self._graph[edge.src][edge.dst][key]["edge"] = existing
                    return
            self._graph.add_edge(edge.src, edge.dst, key=key, edge=edge)

    def remove_edge(self, src: str, dst: str, type: EdgeType) -> None:
        with self._lock:
            key = _edge_key(type)
            if self._graph.has_edge(src, dst, key=key):
                self._graph.remove_edge(src, dst, key=key)

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
        allowed: Optional[set] = None
        if edge_types is not None:
            allowed = {
                et.value if isinstance(et, EdgeType) else EdgeType(et).value
                for et in edge_types
            }
        with self._lock:
            if not self._graph.has_node(node_id):
                return []
            collected: List[MemoryEdge] = []
            seen_edges: set = set()
            frontier: deque = deque([(node_id, 0)])
            visited_nodes = {node_id}
            while frontier:
                current, d = frontier.popleft()
                if d >= depth:
                    continue
                # outgoing
                for _, dst, key, data in self._graph.out_edges(current, keys=True, data=True):
                    self._collect(
                        data, key, current, dst, allowed, min_weight,
                        include_archived, collected, seen_edges,
                    )
                    if dst not in visited_nodes:
                        visited_nodes.add(dst)
                        frontier.append((dst, d + 1))
                # incoming (undirected traversal for semantic neighbourhoods)
                for src, _, key, data in self._graph.in_edges(current, keys=True, data=True):
                    self._collect(
                        data, key, src, current, allowed, min_weight,
                        include_archived, collected, seen_edges,
                    )
                    if src not in visited_nodes:
                        visited_nodes.add(src)
                        frontier.append((src, d + 1))
            return collected

    def _collect(
        self,
        data: dict,
        key: str,
        src: str,
        dst: str,
        allowed: Optional[set],
        min_weight: float,
        include_archived: bool,
        collected: List[MemoryEdge],
        seen_edges: set,
    ) -> None:
        edge: Optional[MemoryEdge] = data.get("edge")
        if edge is None:
            return
        if allowed is not None and edge.type.value not in allowed:
            return
        if edge.weight < min_weight:
            return
        if not include_archived:
            for endpoint in (src, dst):
                node_data = self._graph.nodes.get(endpoint) or {}
                node_obj: Optional[MemoryNode] = node_data.get("node")
                if node_obj is not None and node_obj.archived:
                    return
        sig = (src, dst, key)
        if sig in seen_edges:
            return
        seen_edges.add(sig)
        collected.append(edge)

    def all_edges(self) -> Iterable[MemoryEdge]:
        with self._lock:
            edges: List[MemoryEdge] = []
            for _, _, data in self._graph.edges(data=True):
                edge = data.get("edge")
                if edge is not None:
                    edges.append(edge)
            return edges

    def iter_nodes(self, include_archived: bool = True) -> Iterable[MemoryNode]:
        with self._lock:
            nodes: List[MemoryNode] = []
            for _, data in self._graph.nodes(data=True):
                node: Optional[MemoryNode] = data.get("node")
                if node is None:
                    continue
                if not include_archived and node.archived:
                    continue
                nodes.append(node)
            return nodes

    def edges_for_node(self, node_id: str) -> Iterable[MemoryEdge]:
        with self._lock:
            if not self._graph.has_node(node_id):
                return []
            edges: List[MemoryEdge] = []
            for _, _, data in self._graph.out_edges(node_id, data=True):
                edge = data.get("edge")
                if edge is not None:
                    edges.append(edge)
            for _, _, data in self._graph.in_edges(node_id, data=True):
                edge = data.get("edge")
                if edge is not None:
                    edges.append(edge)
            return edges

    # ---- embeddings ----------------------------------------------------
    def set_embedding(self, node_id: str, vector: List[float]) -> None:
        with self._lock:
            self._embeddings[node_id] = list(vector)

    def get_embedding(self, node_id: str) -> Optional[List[float]]:
        with self._lock:
            vec = self._embeddings.get(node_id)
            return list(vec) if vec is not None else None

    def iter_embeddings(self) -> Iterable[Tuple[str, List[float]]]:
        with self._lock:
            return [(nid, list(vec)) for nid, vec in self._embeddings.items()]

    # ---- persistence ---------------------------------------------------
    def snapshot(self, path: str) -> None:
        with self._lock:
            payload = {
                "version": 1,
                "graph": self._graph,
                "embeddings": self._embeddings,
            }
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        with self._lock:
            self._graph = payload.get("graph", nx.MultiDiGraph())
            self._embeddings = payload.get("embeddings", {})

    # ---- stats ---------------------------------------------------------
    def node_count(self, include_archived: bool = False) -> int:
        with self._lock:
            if include_archived:
                return self._graph.number_of_nodes()
            count = 0
            for _, data in self._graph.nodes(data=True):
                node: Optional[MemoryNode] = data.get("node")
                if node is None or not node.archived:
                    count += 1
            return count

    def edge_count(self) -> int:
        with self._lock:
            return self._graph.number_of_edges()
