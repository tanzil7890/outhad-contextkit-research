""" Graph-first retrieval over the Context-Graph Layer.

Pipeline:

1. **Seed** — accept the top-K vector-store hits as anchor nodes.
2. **Expand** — BFS through the context graph up to ``expansion_depth``,
   propagating a reach score (``accum * edge.weight``) per hop and stopping at
   ``max_candidates``.
3. **Re-rank** — combine the dense score (from the seed search), a cheap
   lexical overlap proxy, and the graph reach score using the weights
   ``alpha_dense`` / ``beta_bm25`` / ``gamma_graph`` from
   :class:`RetrievalConfig`.

The retriever owns no IO. Resolving payloads for BFS-expanded candidates is
delegated to a caller-supplied ``payload_resolver`` so the class stays trivial
to unit-test without booting a vector store.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from outhad_contextkit.memory.context_graph.config import RetrievalConfig
from outhad_contextkit.memory.context_graph.facade import ContextGraph
from outhad_contextkit.memory.context_graph.scoring import (
    cached_frequency_peak,
    normalised_frequency_with_peak,
)
from outhad_contextkit.memory.context_graph.types import EdgeType, MemoryEdge

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

PayloadResolver = Callable[[Sequence[str]], Dict[str, Dict[str, Any]]]


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return {tok.lower() for tok in _TOKEN_RE.findall(text)}


def _lexical_overlap(query_tokens: Set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(text)
    if not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@dataclass
class _Candidate:
    memory_id: str
    reach: float
    dense: float
    lexical: float
    payload: Dict[str, Any]


def _trim_seeds(
    seed_results: Sequence[Dict[str, Any]],
    seed_top_k: int,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    trimmed = sorted(
        (s for s in seed_results if s.get("id")),
        key=lambda r: float(r.get("score") or 0.0),
        reverse=True,
    )[:seed_top_k]
    lookup: Dict[str, Dict[str, Any]] = {s["id"]: s for s in trimmed}
    return lookup, list(lookup.keys())


def _resolve_missing_payloads(
    reach: Dict[str, float],
    seed_lookup: Dict[str, Dict[str, Any]],
    payload_resolver: Optional[PayloadResolver],
) -> Dict[str, Dict[str, Any]]:
    missing_ids = [cid for cid in reach if cid not in seed_lookup]
    resolved: Dict[str, Dict[str, Any]] = {}
    if missing_ids and payload_resolver is not None:
        try:
            fetched = payload_resolver(missing_ids) or {}
            for cid, payload in fetched.items():
                if payload:
                    resolved[cid] = payload
        except Exception as exc:  # pragma: no cover - resolver may fail
            logger.debug("CGL payload_resolver failed: %s", exc)
    return resolved


def _rerank_candidates(
    cg: ContextGraph,
    cfg: RetrievalConfig,
    *,
    query: str,
    seed_lookup: Dict[str, Dict[str, Any]],
    reach: Dict[str, float],
    edges_used: Dict[Tuple[str, str, str], MemoryEdge],
    query_embedding: Optional[Sequence[float]],
    payload_resolver: Optional[PayloadResolver],
    limit: int,
    personal_boost: Optional[Any] = None,
    success_provider: Optional[Any] = None,
    tenant_id: Optional[str] = None,
    sub_tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared rerank used by both BFS and PPR retrievers.

    Pure function over the inputs — no traversal, no backend mutation.
    The only network-like action is ``payload_resolver`` for candidates
    that weren't part of the seed set.

    ``tenant_id`` / ``sub_tenant_id`` are optional hard-skip
    filters. When supplied, candidates whose backing CGL node carries a
    *different* non-NULL tenant tag are excluded — defence in depth on
    top of the storage-level isolation that mode='collection' provides.
    NULL tags are tolerated so legacy nodes still surface.
    """
    resolved_payloads = _resolve_missing_payloads(
        reach, seed_lookup, payload_resolver
    )

    query_tokens = _tokenize(query)
    q_emb = list(query_embedding) if query_embedding else None
    candidates: List[_Candidate] = []
    for cid, reach_score in reach.items():
        payload = seed_lookup.get(cid) or resolved_payloads.get(cid)
        if payload is None:
            payload = {"id": cid, "memory": "", "score": 0.0}
        #  hard tenant/sub-tenant skip. We resolve the backing
        # node lazily so the filter touches the backend at most once per
        # candidate (matches the existing relevance read pattern below).
        if tenant_id is not None or sub_tenant_id is not None:
            node = cg.backend.get_node(cid)
            if node is not None:
                node_tenant = getattr(node, "tenant_id", None)
                if (
                    tenant_id is not None
                    and node_tenant is not None
                    and node_tenant != tenant_id
                ):
                    continue
                node_sub = getattr(node, "sub_tenant_id", None)
                if (
                    sub_tenant_id is not None
                    and node_sub is not None
                    and node_sub != sub_tenant_id
                ):
                    continue
        dense_score = float(payload.get("score") or 0.0)
        if cid not in seed_lookup and q_emb is not None:
            node_vec = cg.get_embedding(cid)
            if node_vec is not None:
                dense_score = max(dense_score, _cosine(q_emb, node_vec))
        lexical_score = _lexical_overlap(query_tokens, payload.get("memory", ""))
        candidates.append(
            _Candidate(
                memory_id=cid,
                reach=float(reach_score),
                dense=dense_score,
                lexical=lexical_score,
                payload=dict(payload),
            )
        )

    #  single peak read up-front keeps the rerank loop O(1) in
    # backend calls per candidate. Cached peak is safe: it would only shift
    # frequency scores by a constant rescale mid-query, which does not
    # change the ranking order within a single search.
    freq_peak = (
        cached_frequency_peak(cg) if cfg.epsilon_frequency > 0 else 0
    )
    frequency_contributions: Dict[str, float] = {}
    #  δ·personal term. The provider caches per-memory boosts
    # internally so re-ranking stays O(N) in SQLite reads (one per id).
    personal_contributions: Dict[str, float] = {}
    personal_enabled = bool(
        personal_boost is not None and cfg.delta_personal > 0
    )
    #  ζ·success term. Provider caches the per-pair hit-rate so
    # repeat lookups inside the rerank cost zero SQLite reads.
    success_contributions: Dict[str, float] = {}
    success_enabled = bool(
        success_provider is not None
        and getattr(cfg, "zeta_success", 0.0) > 0
    )

    def _final_score(c: _Candidate) -> float:
        node = cg.backend.get_node(c.memory_id)
        relevance = 1.0
        freq_term = 0.0
        if node is not None:
            relevance = max(
                0.0, min(1.0, float(getattr(node, "relevance", 1.0)))
            )
            if cfg.epsilon_frequency > 0:
                freq_term = normalised_frequency_with_peak(node, freq_peak)
        frequency_contributions[c.memory_id] = float(freq_term)
        personal_term = 0.0
        if personal_enabled:
            try:
                personal_term = float(personal_boost.boost_for(c.memory_id))
            except Exception:  # pragma: no cover - defensive
                personal_term = 0.0
            # Clamp defensively — provider should already stay in [-1,1].
            personal_term = max(-1.0, min(1.0, personal_term))
        personal_contributions[c.memory_id] = personal_term
        success_term = 0.0
        if success_enabled:
            try:
                success_term = float(success_provider.hit_rate(c.memory_id))
            except Exception:  # pragma: no cover - defensive
                success_term = 0.0
            success_term = max(0.0, min(1.0, success_term))
        success_contributions[c.memory_id] = success_term
        base = (
            cfg.alpha_dense * c.dense
            + cfg.beta_bm25 * c.lexical
            + cfg.gamma_graph * c.reach
            + cfg.epsilon_frequency * freq_term
            + cfg.delta_personal * personal_term
            + getattr(cfg, "zeta_success", 0.0) * success_term
        )
        return base * relevance

    scored: List[Tuple[float, _Candidate]] = [
        (_final_score(c), c) for c in candidates
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    results: List[Dict[str, Any]] = []
    for final, c in scored[: max(1, limit)]:
        entry = dict(c.payload)
        entry.setdefault("id", c.memory_id)
        entry["score"] = float(final)
        entry["context_graph"] = {
            "dense": float(c.dense),
            "lexical": float(c.lexical),
            "graph": float(c.reach),
            "frequency": float(frequency_contributions.get(c.memory_id, 0.0)),
            "personal": float(personal_contributions.get(c.memory_id, 0.0)),
            "success": float(success_contributions.get(c.memory_id, 0.0)),
            "was_seed": c.memory_id in seed_lookup,
        }
        results.append(entry)

    subgraph = [
        {
            "src": edge.src,
            "dst": edge.dst,
            "type": edge.type.value,
            "weight": float(edge.weight),
        }
        for edge in edges_used.values()
    ]

    return {"results": results, "subgraph": subgraph}


class GraphFirstRetriever:
    """Seed → BFS expand → re-rank orchestrator."""

    def __init__(self, context_graph: ContextGraph, config: RetrievalConfig) -> None:
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
        """Return a graph-expanded, re-ranked result list + the used subgraph.

        ``personal_boost``  is an optional ``PersonalBoostProvider``
        that supplies a ``[-1, 1]`` per-memory score. Ignored unless the
        caller also sets ``RetrievalConfig.delta_personal > 0``.

        ``success_provider``  is an optional ``SuccessProvider``
        that supplies a ``[0, 1]`` per-(query_hash, memory) hit-rate.
        Ignored unless the caller also sets ``RetrievalConfig.zeta_success > 0``.

        ``tenant_id`` / ``sub_tenant_id`` hard-skip candidates
        whose backing CGL node carries a different non-NULL tenant tag.
        Defence in depth on top of mode='collection' storage isolation.
        """
        if not seed_results:
            return {"results": [], "subgraph": []}

        cfg = self.config
        seed_top_k = max(1, cfg.seed_top_k)
        expansion_depth = max(0, cfg.expansion_depth)
        edge_floor = cfg.edge_weight_floor
        max_candidates = max(1, cfg.max_candidates)

        seed_lookup, seed_ids = _trim_seeds(seed_results, seed_top_k)

        reach: Dict[str, float] = {sid: 1.0 for sid in seed_ids}
        edges_used: Dict[Tuple[str, str, str], MemoryEdge] = {}
        frontier: List[Tuple[str, float]] = [(sid, 1.0) for sid in seed_ids]
        for _ in range(expansion_depth):
            if not frontier or len(reach) >= max_candidates:
                break
            next_frontier: List[Tuple[str, float]] = []
            for node, accum in frontier:
                try:
                    neighbours = self.cg.backend.neighbours(
                        node,
                        depth=1,
                        min_weight=edge_floor,
                        include_archived=include_archived,
                    )
                except Exception as exc:  # pragma: no cover - backend dependent
                    logger.debug("CGL neighbours() failed for %s: %s", node, exc)
                    neighbours = []
                for edge in neighbours:
                    other = edge.dst if edge.src == node else edge.src
                    if other == node:
                        continue
                    propagated = accum * max(0.0, min(1.0, edge.weight))
                    if propagated <= 0.0:
                        continue
                    current = reach.get(other)
                    if current is None or current < propagated:
                        reach[other] = propagated
                        next_frontier.append((other, propagated))
                    sig = (edge.src, edge.dst, edge.type.value)
                    if sig not in edges_used:
                        edges_used[sig] = edge
                    if len(reach) >= max_candidates:
                        break
                if len(reach) >= max_candidates:
                    break
            frontier = next_frontier

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


__all__ = [
    "GraphFirstRetriever",
    "_rerank_candidates",
    "_trim_seeds",
    "_resolve_missing_payloads",
]
