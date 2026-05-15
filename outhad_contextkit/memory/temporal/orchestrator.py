"""Retrieval orchestrator for TCMGM."""
import logging
from collections import OrderedDict
from threading import RLock
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime

from outhad_contextkit.memory.temporal.types import TimeWindow

logger = logging.getLogger(__name__)


# Phase P6 — process-local LRU for query embeddings. Most TCMGM
# deployments see the same query repeated (auto-complete UIs,
# pagination, A/B traffic). Caching saves 50-200 ms per repeat.
_QUERY_EMBED_CACHE: "OrderedDict[Tuple[int, str, str], Tuple[float, ...]]" = OrderedDict()
_QUERY_EMBED_CACHE_CAP = 1024
_QUERY_EMBED_LOCK = RLock()


def _cached_query_embed(embedding_model: Any, query: str, mode: str) -> List[float]:
    """Return cached or freshly-computed embedding for ``query``.

    Cache key includes ``id(embedding_model)`` so swapping providers
    invalidates automatically. Tuple values are stored to keep the
    cache hashable; converted back to ``list`` for the caller.
    """
    if not query or embedding_model is None:
        return embedding_model.embed(query, mode) if embedding_model else []
    key = (id(embedding_model), query, mode)
    with _QUERY_EMBED_LOCK:
        cached = _QUERY_EMBED_CACHE.get(key)
        if cached is not None:
            _QUERY_EMBED_CACHE.move_to_end(key)
            return list(cached)
    fresh = embedding_model.embed(query, mode)
    try:
        as_tuple = tuple(float(x) for x in fresh)
    except TypeError:
        # Embedding wasn't iterable (rare) — skip cache, return as-is.
        return fresh
    with _QUERY_EMBED_LOCK:
        _QUERY_EMBED_CACHE[key] = as_tuple
        _QUERY_EMBED_CACHE.move_to_end(key)
        while len(_QUERY_EMBED_CACHE) > _QUERY_EMBED_CACHE_CAP:
            _QUERY_EMBED_CACHE.popitem(last=False)
    return list(as_tuple)


def clear_query_embed_cache() -> int:
    """Test helper. Returns count of evicted entries."""
    with _QUERY_EMBED_LOCK:
        n = len(_QUERY_EMBED_CACHE)
        _QUERY_EMBED_CACHE.clear()
        return n


class RetrievalOrchestrator:
    """Orchestrates retrieval across vector, graph, and timeline."""
    
    def __init__(self, vector_store, graph_store, timeline_builder, embedding_model=None):
        """
        Initialize the retrieval orchestrator.
        
        Args:
            vector_store: Vector store instance for semantic search
            graph_store: Graph store instance for entity relationships
            timeline_builder: TimelineBuilder instance for temporal queries
            embedding_model: Embedding model for generating query embeddings
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.timeline = timeline_builder
        self.embedding_model = embedding_model
        logger.info("RetrievalOrchestrator initialized successfully")
    
    def fused_search(
        self,
        query: str,
        user_id: str,
        time_window: Optional[TimeWindow] = None,
        include_causal: bool = True,
        include_multimodal: bool = True,
        top_k: int = 10,
        parallel: bool = True,
        rerank: bool = False,
        rerank_top_k: Optional[int] = None,
    ) -> Dict:
        """
        Fused retrieval combining vector, graph, and timeline.

        Phase P1 — by default the four stages that don't depend on
        the vector seed list (graph, timeline, causal, cross-modal)
        run concurrently inside a ``ThreadPoolExecutor``. Vector search
        always runs first because the causal stage seeds from its
        results. The same call returns the legacy sequential shape
        when ``parallel=False`` (useful for debugging / regression
        gating).

        Args:
            query: Search query
            user_id: User identifier
            time_window: Optional time window filter
            include_causal: Include causal chain exploration
            include_multimodal: Include cross-modal results
            top_k: Number of results
            parallel: When True (default), fan stages 2–5 out via a
                     thread pool. When False, run sequentially.

        Returns:
            Dict with merged results from all retrieval methods.
        """
        logger.info(
            "Starting fused search for query: %r (user: %s, parallel=%s)",
            query,
            user_id,
            parallel,
        )

        results = {
            "vector_results": [],
            "graph_results": [],
            "timeline_results": [],
            "causal_chains": [],
            "multimodal_results": [],
            "fused_ranking": [],
        }

        # 1. Vector search runs first — causal exploration seeds from it.
        logger.debug("Performing vector search...")
        vector_results = self._vector_search(query, user_id, time_window, top_k)
        results["vector_results"] = vector_results
        logger.info(f"Vector search returned {len(vector_results)} results")

        if parallel:
            graph_results, timeline_results, causal_chains, multimodal_results = (
                self._run_stages_parallel(
                    query=query,
                    user_id=user_id,
                    time_window=time_window,
                    include_causal=include_causal,
                    include_multimodal=include_multimodal,
                    vector_results=vector_results,
                )
            )
        else:
            # Legacy sequential path.
            logger.debug("Performing graph search (sequential)...")
            graph_results = self._graph_search(query, user_id, time_window)

            timeline_results: List[Dict] = []
            if time_window:
                logger.debug("Performing timeline search (sequential)...")
                timeline_results = self._timeline_search(
                    user_id, time_window, query=query
                )

            causal_chains: List[Dict] = []
            if include_causal and vector_results:
                logger.debug("Exploring causal chains (sequential)...")
                causal_chains = self._explore_causal_chains(
                    vector_results, user_id
                )

            multimodal_results: List[Dict] = []
            if include_multimodal:
                logger.debug("Performing cross-modal search (sequential)...")
                multimodal_results = self._cross_modal_search(query, user_id)

        results["graph_results"] = graph_results
        results["timeline_results"] = timeline_results
        results["causal_chains"] = causal_chains
        results["multimodal_results"] = multimodal_results
        logger.info(
            "Stage counts: graph=%d timeline=%d causal=%d multimodal=%d",
            len(graph_results),
            len(timeline_results),
            len(causal_chains),
            len(multimodal_results),
        )

        # 6. Fuse and rank all results (pure CPU; not parallelised).
        logger.debug("Fusing and ranking results...")
        fused = self._fuse_results(results, query)

        # Phase A6 — optional cross-encoder rerank. The reranker takes
        # the wider top-(rerank_top_k or 50) bi-encoder set and
        # re-orders by cross-encoder relevance, then the caller's
        # ``top_k`` is applied. Failure (missing dependency, model
        # error) returns the bi-encoder ranking unchanged so retrieval
        # never breaks.
        if rerank and fused:
            try:
                from outhad_contextkit.memory.temporal.reranker import (
                    rerank_results,
                )

                pool_size = int(rerank_top_k or max(50, top_k * 5))
                pool = fused[:pool_size]
                fused = rerank_results(query=query, results=pool)
                logger.info(
                    "Cross-encoder reranked %d → %d candidates",
                    len(pool),
                    len(fused),
                )
            except Exception as exc:  # pragma: no cover - never fatal
                logger.debug("Reranker step failed; using bi-encoder order: %s", exc)

        results["fused_ranking"] = fused[:top_k]
        logger.info(f"Fused ranking complete: {len(results['fused_ranking'])} results")

        return results

    def _run_stages_parallel(
        self,
        *,
        query: str,
        user_id: str,
        time_window: Optional[TimeWindow],
        include_causal: bool,
        include_multimodal: bool,
        vector_results: List[Dict],
    ) -> tuple:
        """Phase P1 — fan the 4 vector-independent stages out concurrently.

        Returns ``(graph_results, timeline_results, causal_chains,
        multimodal_results)``. Each stage is wrapped in its own
        ``try/except`` (matches the legacy per-stage behaviour) so a
        single backend failure cannot abort the whole search.
        """
        from concurrent.futures import ThreadPoolExecutor

        graph_results: List[Dict] = []
        timeline_results: List[Dict] = []
        causal_chains: List[Dict] = []
        multimodal_results: List[Dict] = []

        # max_workers=4 covers the four optional stages; thread pool
        # is short-lived (one search) so no daemon threads survive.
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_graph = ex.submit(
                self._graph_search, query, user_id, time_window
            )
            f_timeline = (
                ex.submit(
                    self._timeline_search, user_id, time_window, query
                )
                if time_window
                else None
            )
            f_causal = (
                ex.submit(self._explore_causal_chains, vector_results, user_id)
                if include_causal and vector_results
                else None
            )
            f_multi = (
                ex.submit(self._cross_modal_search, query, user_id)
                if include_multimodal
                else None
            )

            try:
                graph_results = f_graph.result() or []
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("graph stage failed: %s", exc)

            if f_timeline is not None:
                try:
                    timeline_results = f_timeline.result() or []
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("timeline stage failed: %s", exc)

            if f_causal is not None:
                try:
                    causal_chains = f_causal.result() or []
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("causal stage failed: %s", exc)

            if f_multi is not None:
                try:
                    multimodal_results = f_multi.result() or []
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("multimodal stage failed: %s", exc)

        return graph_results, timeline_results, causal_chains, multimodal_results
    
    def _vector_search(
        self,
        query: str,
        user_id: str,
        time_window: Optional[TimeWindow],
        top_k: int
    ) -> List[Dict]:
        """Perform vector search."""
        try:
            # Generate query embeddings using the embedding model
            if not self.embedding_model:
                logger.warning("No embedding model available for vector search")
                return []
            
            # Phase P6 — cached embedding lookup; identical (model, query,
            # mode) tuples skip the underlying embed call.
            query_embeddings = _cached_query_embed(
                self.embedding_model, query, "search"
            )
            
            # Use the vector store's search method
            filters = {"user_id": user_id}
            
            # CRITICAL FIX: Do NOT filter vector search by time window
            # Vector search should return semantically relevant results regardless of time
            # Temporal filtering is handled by timeline search and in result fusion
            # Filtering by timestamp here causes 0 results because timestamps aren't properly indexed
            
            # Perform search with embeddings (no temporal filtering)
            search_results = self.vector_store.search(
                query=query,
                vectors=query_embeddings,
                filters=filters,
                limit=top_k
            )
            
            # Format results
            formatted_results = []
            for result in search_results:
                if hasattr(result, 'payload'):
                    # Qdrant/Chroma format
                    formatted_results.append({
                        "id": result.id,
                        "content": result.payload.get("data", ""),
                        "score": getattr(result, 'score', 0.0),
                        "metadata": result.payload,
                        "source": "vector"
                    })
                elif isinstance(result, dict):
                    # Dict format (already formatted from Memory.search)
                    formatted_results.append({
                        "id": result.get("id", ""),
                        "content": result.get("memory", ""),
                        "score": result.get("score", 0.0),
                        "metadata": result,
                        "source": "vector"
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _graph_search(
        self,
        query: str,
        user_id: str,
        time_window: Optional[TimeWindow]
    ) -> List[Dict]:
        """Perform graph search."""
        try:
            filters = {"user_id": user_id}
            
            # Use graph's search method
            graph_results = self.graph_store.search(query, filters)
            
            # Format results
            formatted_results = []
            for result in graph_results:
                if isinstance(result, dict):
                    formatted_results.append({
                        "id": result.get("id", ""),
                        "content": result.get("name", ""),
                        "score": 0.8,  # Default relevance score
                        "metadata": result,
                        "source": "graph"
                    })
                else:
                    formatted_results.append({
                        "id": str(result),
                        "content": str(result),
                        "score": 0.8,
                        "metadata": {},
                        "source": "graph"
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    def _timeline_search(
        self,
        user_id: str,
        time_window: TimeWindow,
        query: Optional[str] = None,
    ) -> List[Dict]:
        """Search timeline within time window.

        Phase A5 — when ``query`` is supplied, re-rank time-window
        results by semantic relevance to the query before returning.
        Combines a temporal recency floor (constant 0.7) with cosine
        similarity over the cached query embedding so timeline hits
        are simultaneously time-bounded *and* topic-relevant. Results
        without query rerank fall back to the legacy time-only order.
        """
        try:
            timeline_events = self.timeline.get_timeline(user_id, time_window)
            if not timeline_events:
                return []

            # Format results (legacy shape preserved).
            formatted_results: List[Dict] = []
            for event in timeline_events:
                formatted_results.append({
                    "id": event.get("id", ""),
                    "content": event.get("content", ""),
                    "score": 0.7,  # Default temporal relevance
                    "timestamp": event.get("timestamp", ""),
                    "metadata": event,
                    "source": "timeline",
                })

            # Phase A5 — semantic rerank.
            if query and self.embedding_model is not None:
                try:
                    formatted_results = self._semantic_rerank_timeline(
                        formatted_results, query=query
                    )
                except Exception as exc:  # pragma: no cover - rerank is best-effort
                    logger.debug("Timeline semantic rerank failed: %s", exc)

            return formatted_results
        except Exception as e:
            logger.error(f"Timeline search failed: {e}")
            return []

    def _semantic_rerank_timeline(
        self,
        events: List[Dict],
        *,
        query: str,
        recency_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> List[Dict]:
        """Phase A5 — combine recency + cosine relevance."""
        if not events:
            return events
        # Reuse the same cached embedding helper as _vector_search.
        query_vec = _cached_query_embed(self.embedding_model, query, "search")
        if not query_vec:
            return events

        # Cosine similarity helper. Falls back to simple token overlap
        # when an event has no embedding side-channel (most timeline
        # events come from the SQLite history layer without vectors).
        import math

        def _cosine(a: List[float], b: List[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            num = sum(x * y for x, y in zip(a, b))
            da = math.sqrt(sum(x * x for x in a))
            db = math.sqrt(sum(y * y for y in b))
            return num / (da * db) if da and db else 0.0

        query_tokens = {
            t.lower() for t in (query or "").split() if len(t) >= 3
        }

        for event in events:
            md = event.get("metadata") or {}
            event_vec = (
                md.get("embedding")
                if isinstance(md, dict)
                else None
            )
            if isinstance(event_vec, list) and event_vec:
                semantic = _cosine(query_vec, event_vec)
            else:
                # Token-overlap fallback.
                text = (event.get("content") or "").lower()
                if not query_tokens:
                    semantic = 0.0
                else:
                    text_tokens = {t for t in text.split() if len(t) >= 3}
                    overlap = len(query_tokens & text_tokens)
                    semantic = overlap / max(1, len(query_tokens))
            recency = float(event.get("score", 0.7) or 0.7)
            event["semantic_score"] = float(semantic)
            event["score"] = (
                recency_weight * recency + semantic_weight * semantic
            )

        events.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return events
    
    def _explore_causal_chains(
        self,
        seed_events: List[Dict],
        user_id: str,
        max_depth: int = 3
    ) -> List[Dict]:
        """Explore causal chains from seed events.

        Phase P4 — batched ``UNWIND``-driven query collapses the
        previous N+1 pattern (3 seeds × forward+backward = 6 round-trips)
        into 2 round-trips total. Falls back to the per-event loop on
        backend error so partial outages still return chains.
        """
        from outhad_contextkit.memory.temporal.causal_queries import (
            get_causal_chain,
            get_causal_chains_batch,
        )

        # Top 3 events by score (preserves prior behaviour).
        top_events = [e for e in seed_events[:3] if e.get("id")]
        if not top_events:
            return []

        chains: List[Dict] = []
        try:
            batched = get_causal_chains_batch(
                self.graph_store.graph,
                event_ids=[e["id"] for e in top_events],
                filters={"user_id": user_id},
                max_depth=max_depth,
            )
            for event in top_events:
                eid = event["id"]
                pair = batched.get(eid, {"forward": [], "backward": []})
                chains.append({
                    "seed_event": event,
                    "forward_chain": pair.get("forward", []),
                    "backward_chain": pair.get("backward", []),
                })
            return chains
        except Exception as exc:
            logger.warning(
                "Batched causal exploration failed (%s); "
                "falling back to per-event queries",
                exc,
            )

        # Legacy per-event fallback path — same behaviour as pre-P4.
        for event in top_events:
            event_id = event["id"]
            try:
                forward_chain = get_causal_chain(
                    self.graph_store.graph,
                    event_id,
                    {"user_id": user_id},
                    max_depth=max_depth,
                    direction="forward",
                )
                backward_chain = get_causal_chain(
                    self.graph_store.graph,
                    event_id,
                    {"user_id": user_id},
                    max_depth=max_depth,
                    direction="backward",
                )
                chains.append({
                    "seed_event": event,
                    "forward_chain": forward_chain,
                    "backward_chain": backward_chain,
                })
            except Exception as e:
                logger.error(
                    "Failed to explore causal chain for event %s: %s",
                    event_id,
                    e,
                )

        return chains
    
    def _cross_modal_search(
        self,
        query: str,
        user_id: str
    ) -> List[Dict]:
        """Perform cross-modal search."""
        try:
            from outhad_contextkit.memory.temporal.cross_modal import cross_modal_search
            
            # Get query embedding
            # This would use the multimodal embedder if available
            # For now, return empty as this requires multimodal content in the database
            
            # TODO: Implement full cross-modal search when multimodal content is available
            # This would:
            # 1. Get query embedding
            # 2. Query for nodes with modality != 'text'
            # 3. Perform cross-modal similarity search
            
            logger.debug("Cross-modal search placeholder - no multimodal content indexed")
            return []
        except Exception as e:
            logger.error(f"Cross-modal search failed: {e}")
            return []
    
    def _fuse_results(
        self,
        all_results: Dict,
        query: str,
        *,
        method: str = "rrf",
        rrf_k: int = 60,
    ) -> List[Dict]:
        """Fuse and rank results from all sources.

        Phase A1 — default fusion is **Reciprocal Rank Fusion (RRF)**:
        ``score(item) = Σ 1 / (k + rank_in_channel)`` with ``k=60``.
        RRF is the industry standard for hybrid retrieval (Elastic,
        Vespa, Pinecone) because it normalises across channels with
        incomparable score scales (cosine in [-1, 1] vs graph reach in
        [0, 1] vs timeline recency in seconds). Pass ``method="weighted"``
        to fall back to the legacy per-channel weighted-sum (kept for
        regression gating + comparison).

        Args:
            all_results: Dict with keys ``vector_results`` /
                ``graph_results`` / ``timeline_results``.
            query: Original query (reserved for future query-aware
                tie-breakers; unused today).
            method: ``"rrf"`` (default) or ``"weighted"`` (legacy).
            rrf_k: RRF dampening constant. 60 is the canonical default.

        Returns:
            Deduplicated list ordered by ``final_score`` descending.
        """
        if method == "weighted":
            return self._fuse_weighted(all_results)
        return self._fuse_rrf(all_results, k=rrf_k)

    def _fuse_rrf(self, all_results: Dict, *, k: int = 60) -> List[Dict]:
        """RRF fusion. ``score = Σ 1/(k + rank_i)`` per channel."""
        # Per-channel weights (tunable). Weights bias channels relative
        # to each other while RRF handles within-channel normalisation.
        channel_weights = {
            "vector": 1.0,
            "graph": 0.7,
            "timeline": 0.6,
        }

        # rrf_scores[id] aggregates per-channel contributions.
        rrf_scores: Dict[str, float] = {}
        first_seen: Dict[str, Dict[str, Any]] = {}
        per_id_sources: Dict[str, set] = {}

        def _ingest(channel: str, results: List[Dict]) -> None:
            weight = channel_weights.get(channel, 0.5)
            for rank, result in enumerate(results, start=1):
                rid = result.get("id") or result.get("content") or ""
                if not rid:
                    continue
                rrf_scores[rid] = rrf_scores.get(rid, 0.0) + weight * (
                    1.0 / (k + rank)
                )
                per_id_sources.setdefault(rid, set()).add(channel)
                # Keep the first-seen formatted record so we can return
                # rich metadata (matches the pre-A1 contract).
                if rid not in first_seen:
                    first_seen[rid] = {
                        "content": result.get("content", ""),
                        "id": result.get("id", rid),
                        "source": channel,
                        "score": float(result.get("score", 0.0)),
                        "metadata": dict(result.get("metadata", {}) or {}),
                    }
                    if channel == "timeline":
                        first_seen[rid]["timestamp"] = result.get("timestamp", "")

        _ingest("vector", all_results.get("vector_results", []) or [])
        _ingest("graph", all_results.get("graph_results", []) or [])
        _ingest("timeline", all_results.get("timeline_results", []) or [])

        fused: List[Dict] = []
        for rid, score in rrf_scores.items():
            entry = first_seen[rid]
            entry["final_score"] = float(score)
            entry["fusion_sources"] = sorted(per_id_sources.get(rid, set()))
            entry["fusion_method"] = "rrf"
            fused.append(entry)

        fused.sort(key=lambda x: x["final_score"], reverse=True)

        # Content-level dedup as a safety net (multiple ids with the
        # same content text get collapsed; rare but seen in practice).
        seen_content = set()
        deduplicated: List[Dict] = []
        for item in fused:
            content = item.get("content", "")
            if content and content in seen_content:
                continue
            if content:
                seen_content.add(content)
            deduplicated.append(item)
        return deduplicated

    def _fuse_weighted(self, all_results: Dict) -> List[Dict]:
        """Legacy weighted-sum fusion. Kept for parity / debugging."""
        fused: List[Dict] = []
        # Vector
        for result in all_results.get("vector_results", []) or []:
            fused.append({
                "content": result.get("content", ""),
                "id": result.get("id", ""),
                "source": "vector",
                "score": result.get("score", 0.0),
                "weight": 0.4,
                "metadata": result.get("metadata", {}),
            })
        # Graph
        for result in all_results.get("graph_results", []) or []:
            fused.append({
                "content": result.get("content", ""),
                "id": result.get("id", ""),
                "source": "graph",
                "score": 0.7,
                "weight": 0.3,
                "metadata": result.get("metadata", {}),
            })
        # Timeline
        for result in all_results.get("timeline_results", []) or []:
            fused.append({
                "content": result.get("content", ""),
                "id": result.get("id", ""),
                "source": "timeline",
                "score": 0.6,
                "weight": 0.3,
                "timestamp": result.get("timestamp", ""),
                "metadata": result.get("metadata", {}),
            })
        for item in fused:
            item["final_score"] = item["score"] * item["weight"]
            item["fusion_method"] = "weighted"
        fused.sort(key=lambda x: x["final_score"], reverse=True)
        seen_content = set()
        deduplicated: List[Dict] = []
        for item in fused:
            content = item.get("content", "")
            if content and content in seen_content:
                continue
            if content:
                seen_content.add(content)
            deduplicated.append(item)
        return deduplicated

