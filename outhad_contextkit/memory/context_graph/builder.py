"""Incremental edge synthesis for the Context-Graph Layer.

The builder is called by the hook sites in :class:`Memory` / :class:`AsyncMemory`
after a memory is created, updated, or deleted. It is intentionally
deterministic and side-effect-safe: if an edge cannot be synthesised (e.g. no
embedding available), the hook silently returns rather than raising.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from outhad_contextkit.memory.context_graph.budget import LLMBudget
from outhad_contextkit.memory.context_graph.config import EdgeSynthesisConfig
from outhad_contextkit.memory.context_graph.facade import ContextGraph
from outhad_contextkit.memory.context_graph.llm_edges import (
    InferredEdge,
    LLMEdgeSynthesizer,
)
from outhad_contextkit.memory.context_graph.types import EdgeType

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)
_DOC_ID_KEYS = ("doc_id", "document_id", "source_id", "source_url", "url")


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


def _extract_doc_refs(text: str, metadata: Optional[Dict[str, Any]]) -> List[str]:
    refs: List[str] = []
    for key in _DOC_ID_KEYS:
        value = (metadata or {}).get(key)
        if isinstance(value, str) and value:
            refs.append(value)
    if text:
        refs.extend(_URL_PATTERN.findall(text))
    # De-duplicate while preserving order.
    seen = set()
    out: List[str] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


class IncrementalGraphBuilder:
    """Synthesises structural edges when memories are created or updated."""

    def __init__(
        self,
        context_graph: ContextGraph,
        config: EdgeSynthesisConfig,
        *,
        llm: Optional[Any] = None,
        llm_budget: Optional[LLMBudget] = None,
    ) -> None:
        self.cg = context_graph
        self.config = config
        # Track per-run last-seen memory to wire REPLY_TO / TEMPORAL_NEXT edges.
        self._last_by_run: Dict[Tuple[str, str], Tuple[str, datetime]] = {}
        # Remember doc refs so future memories can link back.
        self._doc_refs: Dict[str, List[str]] = {}
        # Phase C — optional LLM edge synthesiser. Lives here (not in the
        # facade) because the facade owns storage, not inference.
        self._llm = llm
        self._llm_budget = llm_budget or LLMBudget(
            daily_token_budget=getattr(self.config.llm, "daily_token_budget", 0),
            cooldown_seconds=getattr(self.config.llm, "cooldown_seconds", 900),
        )
        self._llm_synth: Optional[LLMEdgeSynthesizer] = None
        if self._llm is not None and (
            self.config.use_llm_inference or self.config.llm.enabled
        ):
            self._llm_synth = LLMEdgeSynthesizer(
                llm=self._llm,
                cfg=self.config.llm,
                budget=self._llm_budget,
            )
        # Cache of node text by id so the LLM synth doesn't re-fetch from
        # the vector store. Populated opportunistically on add/update.
        self._text_by_id: Dict[str, str] = {}

    # ---- public entry points ------------------------------------------
    def on_memory_created(
        self,
        memory_id: str,
        text: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        created_at: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ) -> None:
        self.cg.upsert_memory_node(
            memory_id,
            text,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
            embedding=embedding,
            tenant_id=tenant_id,
            sub_tenant_id=sub_tenant_id,
        )
        now = created_at or datetime.utcnow()
        self._remember_text(memory_id, text)

        self._maybe_reply_to(memory_id, user_id, agent_id, run_id, now)
        self._maybe_temporal_next(memory_id, user_id, agent_id, run_id, now)
        self._maybe_document_link(memory_id, text, metadata)
        topic_scores = self._maybe_topic_similar(memory_id, embedding)
        self._maybe_llm_edges(
            memory_id, text, topic_scores, user_id=user_id
        )

        if run_id or agent_id or user_id:
            key = (run_id or agent_id or "", user_id or "")
            self._last_by_run[key] = (memory_id, now)

    def on_memory_updated(
        self,
        memory_id: str,
        text: str,
        *,
        prev_version_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ) -> None:
        self.cg.upsert_memory_node(
            memory_id,
            text,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
            embedding=embedding,
            tenant_id=tenant_id,
            sub_tenant_id=sub_tenant_id,
        )
        self._remember_text(memory_id, text)
        if prev_version_id and prev_version_id != memory_id:
            self.cg.upsert_edge(
                prev_version_id,
                memory_id,
                EdgeType.UPDATED_FROM,
                weight=1.0,
                user_id=user_id,
                metadata={"reason": "memory_update"},
            )
        topic_scores = self._maybe_topic_similar(memory_id, embedding)
        self._maybe_document_link(memory_id, text, metadata)
        self._maybe_llm_edges(
            memory_id, text, topic_scores, user_id=user_id
        )

    def on_memory_deleted(
        self,
        memory_id: str,
        *,
        user_id: Optional[str] = None,
        hard: bool = False,
    ) -> None:
        if hard:
            self.cg.delete_memory_node(memory_id, user_id=user_id)
        else:
            self.cg.archive_memory_node(memory_id, user_id=user_id)

    # ---- helpers -------------------------------------------------------
    def _maybe_reply_to(
        self,
        memory_id: str,
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        now: datetime,
    ) -> None:
        if not self.config.enable_reply_to:
            return
        key = (run_id or agent_id or "", user_id or "")
        prev = self._last_by_run.get(key)
        if not prev:
            return
        prev_id, prev_ts = prev
        if prev_id == memory_id:
            return
        window = timedelta(seconds=self.config.reply_to_window_seconds)
        if now - prev_ts <= window:
            self.cg.upsert_edge(
                prev_id,
                memory_id,
                EdgeType.REPLY_TO,
                weight=1.0,
                user_id=user_id,
                metadata={"window_s": self.config.reply_to_window_seconds},
            )

    def _maybe_temporal_next(
        self,
        memory_id: str,
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        now: datetime,
    ) -> None:
        if not self.config.enable_temporal_next:
            return
        key = (run_id or agent_id or "", user_id or "")
        prev = self._last_by_run.get(key)
        if not prev:
            return
        prev_id, _ = prev
        if prev_id == memory_id:
            return
        self.cg.upsert_edge(
            prev_id,
            memory_id,
            EdgeType.TEMPORAL_NEXT,
            weight=0.6,
            user_id=user_id,
            metadata={"session_key": f"{key[0]}::{key[1]}"},
        )

    def _maybe_document_link(
        self,
        memory_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if not self.config.enable_document_link:
            return
        refs = _extract_doc_refs(text, metadata)
        if not refs:
            return
        for ref in refs:
            previous_holders = self._doc_refs.get(ref, [])
            for other_id in previous_holders:
                if other_id == memory_id:
                    continue
                self.cg.upsert_edge(
                    memory_id,
                    other_id,
                    EdgeType.DOCUMENT_LINK,
                    weight=0.8,
                    evidence=ref,
                    metadata={"ref": ref},
                )
            if memory_id not in previous_holders:
                self._doc_refs.setdefault(ref, []).append(memory_id)

    def _maybe_topic_similar(
        self,
        memory_id: str,
        embedding: Optional[List[float]],
    ) -> List[Tuple[str, float]]:
        """Persist top-K TOPIC_SIMILAR edges and return the ranked candidates.

        The returned list is used by :meth:`_maybe_llm_edges` as the pool
        of pairs considered for LLM classification. Callers should treat
        it as read-only.
        """
        if not self.config.enable_topic_similar or not embedding:
            return []
        top_k = self.config.topic_top_k
        threshold = self.config.topic_min_similarity
        scores: List[Tuple[str, float]] = []
        for other_id, vec in self.cg.iter_embeddings():
            if other_id == memory_id:
                continue
            score = _cosine(embedding, vec)
            if score >= threshold:
                scores.append((other_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        top = scores[:top_k]
        for other_id, score in top:
            self.cg.upsert_edge(
                memory_id,
                other_id,
                EdgeType.TOPIC_SIMILAR,
                weight=float(score),
                metadata={"similarity": float(score)},
            )
        return top

    def _maybe_llm_edges(
        self,
        memory_id: str,
        text: str,
        topic_scores: Sequence[Tuple[str, float]],
        *,
        user_id: Optional[str],
    ) -> List[InferredEdge]:
        """Call :class:`LLMEdgeSynthesizer` on the post-topic candidates.

        Fail-soft: synthesiser errors are logged and do not propagate. The
        returned list is only useful for tests; persistence happens inline.
        """
        if self._llm_synth is None or not topic_scores:
            return []
        candidates: List[Tuple[str, str, float]] = []
        for cid, sim in topic_scores:
            cand_text = self._text_by_id.get(cid)
            if not cand_text:
                # We only LLM-classify pairs whose text we have locally; a
                # vector-store round-trip on the hot insert path would be
                # unacceptable latency.
                continue
            candidates.append((cid, cand_text, float(sim)))
        if not candidates:
            return []
        try:
            edges = self._llm_synth.infer_edges(
                memory_id, text, candidates, user_id=user_id
            )
        except Exception as exc:  # pragma: no cover - fail-soft
            logger.debug("CGL LLM edge synth raised: %s", exc)
            return []
        for edge in edges:
            self.cg.upsert_edge(
                edge.src,
                edge.dst,
                edge.type,
                weight=edge.weight,
                evidence=edge.evidence,
                user_id=user_id,
                metadata={
                    "source": "llm",
                    "label": edge.type.value,
                    "confidence": float(edge.confidence),
                },
            )
        return edges

    # ---- text cache ----------------------------------------------------
    def _remember_text(self, memory_id: str, text: str) -> None:
        """Opportunistic cache so LLM synth can form candidate prompts."""
        if not memory_id or not text:
            return
        # Bound the cache so long-lived workers don't grow it without limit.
        if len(self._text_by_id) > 2048:
            for stale_id in list(self._text_by_id)[:512]:
                self._text_by_id.pop(stale_id, None)
        self._text_by_id[memory_id] = text

    # ---- maintenance ---------------------------------------------------
    def reset_session_cache(self) -> None:
        """Clear the per-run last-memory cache (used by long-running workers)."""
        self._last_by_run.clear()
        self._doc_refs.clear()
        self._text_by_id.clear()
