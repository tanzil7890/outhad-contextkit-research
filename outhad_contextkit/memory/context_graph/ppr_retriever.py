"""Phase D — Personalised PageRank retrieval over the Context-Graph.

Alternative to :class:`GraphFirstRetriever` that weighs candidates by
their **global** random-walk probability given the seed distribution,
instead of a single-path BFS weight product. Activated by
``RetrievalConfig.algorithm = "ppr"``.

Algorithm:

1. Trim the seed list to ``seed_top_k`` by dense score.
2. Build an in-memory directed graph of edges with
   ``weight >= edge_weight_floor`` (skipping archived nodes unless
   ``include_archived`` is True).
3. Personalised vector = dense score per seed (uniform if scores absent).
4. Run :func:`networkx.pagerank` with ``alpha=ppr_damping``,
   ``max_iter=ppr_max_iter``, ``tol=ppr_tolerance``, weighted by
   ``weight``.
5. Feed the PPR distribution as the ``reach`` signal into the shared
   :func:`_rerank_candidates` helper. The retriever contract
   (``α·dense + β·bm25 + γ·graph`` multiplied by ``relevance``) is
   unchanged — only the "graph" channel's source swaps.

Notes:
* For very large graphs (> 100k nodes) the in-proc PPR is the wrong tool;
  use a server-side GDS pipeline. The retriever logs WARNING once per
  call if the live graph exceeds ``_LARGE_GRAPH_THRESHOLD``.
* NetworkX is required (already imported by other CGL modules).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from outhad_contextkit.memory.context_graph.config import RetrievalConfig
from outhad_contextkit.memory.context_graph.facade import ContextGraph
from outhad_contextkit.memory.context_graph.retriever import (
    PayloadResolver,
    _rerank_candidates,
    _trim_seeds,
)
from outhad_contextkit.memory.context_graph.types import MemoryEdge

logger = logging.getLogger(__name__)

_LARGE_GRAPH_THRESHOLD = 100_000


class PersonalisedPageRankRetriever:
    """Seed → global PPR reach → re-rank orchestrator."""

    def __init__(
        self,
        context_graph: ContextGraph,
        config: RetrievalConfig,
    ) -> None:
        self.cg = context_graph
        self.config = config

    def retrieve(
        self,
        query: str,
        seed_results: List[Dict[str, Any]],
        *,
        limit: int,
        payload_resolver: Optional[PayloadResolver] = None,
        query_embedding: Optional[Sequence[float]] = None,
        include_archived: bool = False,
        personal_boost: Optional[Any] = None,
        success_provider: Optional[Any] = None,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not seed_results:
            return {"results": [], "subgraph": []}

        cfg = self.config
        seed_top_k = max(1, cfg.seed_top_k)
        edge_floor = cfg.edge_weight_floor
        max_candidates = max(1, cfg.max_candidates)

        seed_lookup, seed_ids = _trim_seeds(seed_results, seed_top_k)
        if not seed_ids:
            return {"results": [], "subgraph": []}

        view, edges_used = self._build_graph_view(
            edge_floor=edge_floor, include_archived=include_archived
        )

        # Seeds may be absent from the graph view (e.g. no backing node
        # yet). Register them so PPR can still start from their
        # personalisation mass.
        for sid in seed_ids:
            if sid not in view:
                view.add_node(sid)

        personalisation = self._build_personalisation(seed_lookup, view)
        ppr_scores = self._run_pagerank(view, personalisation)

        # Keep only seeds + top (max_candidates - len(seeds)) by score.
        reach: Dict[str, float] = {sid: 1.0 for sid in seed_ids}
        non_seed = [
            (nid, score)
            for nid, score in ppr_scores.items()
            if nid not in reach
        ]
        non_seed.sort(key=lambda item: item[1], reverse=True)
        remaining = max(0, max_candidates - len(reach))
        for nid, score in non_seed[:remaining]:
            if score <= 0:
                continue
            reach[nid] = float(score)

        return _rerank_candidates(
            self.cg,
            cfg,
            query=query,
            seed_lookup=seed_lookup,
            reach=reach,
            edges_used=edges_used,
            query_embedding=query_embedding,
            payload_resolver=payload_resolver,
            limit=limit,
            personal_boost=personal_boost,
            success_provider=success_provider,
            tenant_id=tenant_id,
            sub_tenant_id=sub_tenant_id,
        )

    # ---- helpers -------------------------------------------------------
    def _build_graph_view(
        self,
        *,
        edge_floor: float,
        include_archived: bool,
    ) -> Tuple[Any, Dict[Tuple[str, str, str], MemoryEdge]]:
        try:
            import networkx as nx  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "networkx is required for PersonalisedPageRankRetriever"
            ) from exc

        g: "nx.DiGraph" = nx.DiGraph()
        edges_used: Dict[Tuple[str, str, str], MemoryEdge] = {}

        skipped_archived: set = set()
        if not include_archived:
            try:
                for node in self.cg.backend.iter_nodes(include_archived=True):
                    if getattr(node, "archived", False):
                        skipped_archived.add(node.id)
            except Exception as exc:  # pragma: no cover - backend dependent
                logger.debug("PPR iter_nodes failed: %s", exc)

        edge_count = 0
        for edge in self.cg.backend.all_edges():
            if edge.weight < edge_floor:
                continue
            if edge.src in skipped_archived or edge.dst in skipped_archived:
                continue
            # Aggregate duplicate edges between the same src/dst by taking
            # the strongest weight — PPR works on a simple weighted digraph.
            existing = g.get_edge_data(edge.src, edge.dst)
            weight = float(edge.weight)
            if existing is None or weight > existing.get("weight", 0.0):
                g.add_edge(edge.src, edge.dst, weight=weight)
            sig = (edge.src, edge.dst, edge.type.value)
            if sig not in edges_used:
                edges_used[sig] = edge
            edge_count += 1

        if g.number_of_nodes() >= _LARGE_GRAPH_THRESHOLD:
            logger.warning(
                "Context-graph has %d nodes; in-proc PPR may be slow. "
                "Consider retrieval.algorithm='bfs' or a server-side GDS pipeline.",
                g.number_of_nodes(),
            )
        logger.debug(
            "PPR graph view built: nodes=%d edges=%d (from %d raw edges)",
            g.number_of_nodes(),
            g.number_of_edges(),
            edge_count,
        )
        return g, edges_used

    def _build_personalisation(
        self,
        seed_lookup: Dict[str, Dict[str, Any]],
        view: Any,
    ) -> Dict[str, float]:
        """Build a personalisation vector biased by seed dense scores.

        Uniform fallback when every seed has score ≤ 0 (e.g. manual seeds
        without a prior vector-store hit).
        """
        raw: Dict[str, float] = {}
        total = 0.0
        for sid, payload in seed_lookup.items():
            score = max(0.0, float(payload.get("score") or 0.0))
            raw[sid] = score
            total += score
        if total <= 0.0:
            # Uniform over seeds, zero on everything else.
            n = len(seed_lookup) or 1
            return {sid: 1.0 / n for sid in seed_lookup}
        # Normalise. Keys missing from ``view`` are added by the caller.
        return {sid: raw[sid] / total for sid in seed_lookup}

    def _run_pagerank(
        self,
        view: Any,
        personalisation: Dict[str, float],
    ) -> Dict[str, float]:
        try:
            import networkx as nx  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "networkx is required for PersonalisedPageRankRetriever"
            ) from exc

        if view.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(
                view,
                alpha=self.config.ppr_damping,
                personalization=personalisation,
                max_iter=self.config.ppr_max_iter,
                tol=self.config.ppr_tolerance,
                weight="weight",
            )
        except Exception as exc:  # pragma: no cover - degenerate graph
            logger.debug("networkx.pagerank failed: %s; falling back to seeds", exc)
            return dict(personalisation)


__all__ = ["PersonalisedPageRankRetriever"]
