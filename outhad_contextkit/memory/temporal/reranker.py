""" optional cross-encoder reranker.

Two-stage retrieval is the industry standard for production RAG:
1. **Bi-encoder** (CLIP / OpenAI / SigLIP) returns top-50 by cosine
   similarity over independently-encoded vectors. Cheap, ANN-able.
2. **Cross-encoder** rescoring of the top-50 to top-10. The
   cross-encoder takes ``[query, candidate]`` together as one input
   and outputs a relevance score, capturing fine-grained interaction
   that bi-encoders miss. Models like ``cross-encoder/ms-marco-MiniLM-L-6-v2``
   (~80MB) deliver +25-35% NDCG@5 over bi-encoder-only at the cost of
   ~30-100 ms per call on CPU.

The reranker is **opt-in**:

* ``sentence-transformers`` is NOT a hard dependency of this package.
* :func:`get_reranker` raises a clear ``RerankerUnavailable`` exception
  with install instructions when the package is missing.
* ``rerank_results`` ignores cross-encoder errors and returns the
  original ranking, so a misconfigured reranker can never break the
  retrieval path.

Default behaviour:
* Module-level RLock-guarded cache so the model loads once per process.
* ``get_reranker()`` returns ``None`` instead of raising when ``strict=False``
  so callers can do ``if r is not None: rerank_results(...)``.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerUnavailable(RuntimeError):
    """Raised when the cross-encoder dependency is missing."""


_CACHE: Dict[str, Any] = {}
_LOCK = threading.RLock()


def get_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    *,
    strict: bool = False,
) -> Optional[Any]:
    """Return a cached ``CrossEncoder`` instance or ``None``.

    First call loads + caches the model (~80MB on disk for the default
    `ms-marco-MiniLM-L-6-v2`). Subsequent calls hit the dict.

    Args:
        model_name: HuggingFace cross-encoder repo id.
        strict: When True, raise :class:`RerankerUnavailable` instead of
            returning ``None`` if ``sentence-transformers`` is missing.
    """
    cached = _CACHE.get(model_name)
    if cached is not None:
        return cached
    with _LOCK:
        cached = _CACHE.get(model_name)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            msg = (
                "Cross-encoder reranker requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
            if strict:
                raise RerankerUnavailable(msg) from exc
            logger.info("%s — reranker disabled, falling back to bi-encoder ordering", msg)
            return None
        try:
            model = CrossEncoder(model_name)
        except Exception as exc:  # pragma: no cover - download / network
            logger.error("Failed to load reranker %s: %s", model_name, exc)
            if strict:
                raise
            return None
        _CACHE[model_name] = model
        logger.info("Reranker loaded: %s", model_name)
        return model


def clear_reranker_cache() -> int:
    """Test helper. Returns evicted entry count."""
    with _LOCK:
        n = len(_CACHE)
        _CACHE.clear()
        return n


def rerank_results(
    query: str,
    results: List[Dict],
    *,
    top_k: Optional[int] = None,
    content_key: str = "content",
    score_key: str = "rerank_score",
    final_score_key: str = "final_score",
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> List[Dict]:
    """Rerank ``results`` via cross-encoder ``(query, candidate)`` scoring.

    On any failure (missing dependency, model load error, inference
    error), returns ``results`` unchanged so retrieval never breaks.

    Args:
        query: Original user query.
        results: List of result dicts from the bi-encoder stage. Each
            must contain ``content_key``.
        top_k: Optional final cap. None → return all reranked.
        content_key: Dict key holding candidate text.
        score_key: New key written with the cross-encoder score.
        final_score_key: When present, also overwritten with the
            cross-encoder score so downstream sort-by-final_score works
            without further plumbing.
        model_name: HuggingFace cross-encoder id.
    """
    if not results or not query:
        return list(results)

    model = get_reranker(model_name, strict=False)
    if model is None:
        return list(results)

    pairs = [(query, str(r.get(content_key, "")) or "") for r in results]
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as exc:  # pragma: no cover - inference error
        logger.warning("Cross-encoder predict failed: %s", exc)
        return list(results)

    enriched: List[Dict] = []
    for r, s in zip(results, scores):
        out = dict(r)
        out[score_key] = float(s)
        out[final_score_key] = float(s)
        enriched.append(out)

    enriched.sort(key=lambda x: x.get(score_key, 0.0), reverse=True)
    if top_k is not None and top_k > 0:
        enriched = enriched[: int(top_k)]
    return enriched


__all__ = [
    "DEFAULT_RERANKER_MODEL",
    "RerankerUnavailable",
    "clear_reranker_cache",
    "get_reranker",
    "rerank_results",
]
