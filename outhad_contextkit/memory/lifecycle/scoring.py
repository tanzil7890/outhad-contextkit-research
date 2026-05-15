"""Phase D2 — decay-score computation.

Combines three signals already tracked on every CGL node into a single
``[0, 1]`` float:

* **Recency** — ``exp(-λ · Δh)`` over hours since ``last_accessed_at``
  (or ``updated_at`` when unset). λ derived from the existing CGL
  ``DecayConfig.half_life_days``.
* **Frequency** — ``log1p(access_count) / log1p(peak_access)``.
* **Feedback** — ``tanh((helpful - unhelpful) / 3)`` remapped to ``[0, 1]``.

The terms are linearly combined with operator-tunable weights from
:class:`DecayV2Config`. Coefficients sum to ≤ 1.0 (validated at config
load time) so the result stays in ``[0, 1]``.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from outhad_contextkit.memory.context_graph.types import MemoryNode
from outhad_contextkit.memory.lifecycle.config import DecayV2Config


def _ensure_aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def compute_decay_score(
    node: MemoryNode,
    *,
    cfg: DecayV2Config,
    peak_access: int,
    half_life_days: float,
    now: Optional[datetime] = None,
) -> float:
    """Return a decay score in ``[0, 1]`` for ``node``.

    ``half_life_days`` is the recency term's half-life — typically wired
    from ``MemoryConfig.context_graph.decay.half_life_days`` so the
    score honours the existing CGL decay clock.
    ``peak_access`` is the maximum access_count across all live nodes,
    used to normalise the frequency term.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    else:
        now = _ensure_aware(now)

    # Recency term (exp(-λ · Δh)).
    last = _ensure_aware(node.last_accessed_at) or _ensure_aware(node.updated_at)
    if last is None:
        recency = 0.0
    else:
        delta_h = max(0.0, (now - last).total_seconds() / 3600.0)
        half_life_h = max(1e-6, float(half_life_days) * 24.0)
        lam = math.log(2.0) / half_life_h
        recency = math.exp(-lam * delta_h)

    # Frequency term (log1p ratio).
    count = int(getattr(node, "access_count", 0) or 0)
    peak = max(0, int(peak_access or 0))
    if count <= 0 or peak <= 0:
        frequency = 0.0
    else:
        frequency = math.log1p(count) / math.log1p(peak)
        frequency = max(0.0, min(1.0, frequency))

    # Feedback term (tanh remapped to [0, 1]).
    helpful = int(getattr(node, "helpful_count", 0) or 0)
    unhelpful = int(getattr(node, "unhelpful_count", 0) or 0)
    if helpful == 0 and unhelpful == 0:
        feedback = 0.5  # neutral when no verdicts yet
    else:
        feedback = (math.tanh((helpful - unhelpful) / 3.0) + 1.0) / 2.0

    score = (
        cfg.weight_recency * recency
        + cfg.weight_frequency * frequency
        + cfg.weight_feedback * feedback
    )
    return max(0.0, min(1.0, score))


def cached_peak_access(cg) -> int:
    """Helper for callers that want to avoid recomputing the peak across
    every score query. Backends ship ``max_access_count`` already (F2).
    """
    try:
        return int(cg.backend.max_access_count() or 0)
    except Exception:
        return 0


__all__ = ["compute_decay_score", "cached_peak_access"]
