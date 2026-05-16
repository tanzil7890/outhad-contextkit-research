"""Scoring helpers shared by the Context-Graph retrievers.

Pure functions only — no IO, no state. Isolating these lets the retriever
layers stay small and lets plug new signals into the final
rerank formula without rewriting :mod:`retriever`.
"""
from __future__ import annotations

import math
from typing import Any, Optional


def normalised_frequency(node: Any, cg: Any) -> float:
    """Log-scaled ``access_count`` normalised to ``[0, 1]``.

    Formula: ``log1p(node.access_count) / log1p(max_access_count)``.

    Returns ``0.0`` when the node has never been accessed or when the
    backend reports a zero peak (cold start).
    """
    count = int(getattr(node, "access_count", 0) or 0)
    if count <= 0:
        return 0.0
    try:
        peak = int(cg.backend.max_access_count() or count)
    except Exception:  # pragma: no cover - backend dependent
        peak = count
    if peak <= 0:
        return 0.0
    if peak < count:
        peak = count
    numerator = math.log1p(count)
    denominator = math.log1p(peak)
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def cached_frequency_peak(cg: Any) -> int:
    """Convenience single-call peak read used by rerankers."""
    try:
        return int(cg.backend.max_access_count() or 0)
    except Exception:  # pragma: no cover - backend dependent
        return 0


def normalised_frequency_with_peak(node: Any, peak: int) -> float:
    """Same as :func:`normalised_frequency` but uses a precomputed peak
    to avoid N backend calls inside a rerank loop.
    """
    count = int(getattr(node, "access_count", 0) or 0)
    if count <= 0:
        return 0.0
    effective_peak = int(peak or 0)
    if effective_peak <= 0:
        effective_peak = count
    if effective_peak < count:
        effective_peak = count
    numerator = math.log1p(count)
    denominator = math.log1p(effective_peak)
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


__all__ = [
    "normalised_frequency",
    "normalised_frequency_with_peak",
    "cached_frequency_peak",
]
