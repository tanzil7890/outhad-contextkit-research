"""Read-only providers consumed by the rerank loop — Phase F4.

A ``PersonalBoostProvider`` looks up a bounded, cached per-user boost
for one memory id so ``_rerank_candidates`` stays a pure function of its
inputs. The provider is constructed once per search and cached in a
dict, so a rerank loop over N candidates performs at most N SQLite
reads (one per candidate), with identical ids short-circuiting through
the cache.

Why a provider instead of passing the store directly? So Phase F8's
pipeline can swap in a fake / pre-computed map for testing without
booting SQLite.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class _FeedbackStoreLike(Protocol):  # pragma: no cover - typing only
    def personal_boost(
        self, memory_id: str, *, user_id: Optional[str] = None
    ) -> float: ...


class PersonalBoostProvider:
    """Bounded per-memory boost lookup keyed by user id.

    The class intentionally does not know about ``FeedbackStore``
    internals — any object with a ``personal_boost`` method works.
    """

    def __init__(
        self,
        feedback_store: _FeedbackStoreLike,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self._store = feedback_store
        self._user_id = user_id
        self._agent_id = agent_id
        self._run_id = run_id
        self._cache: Dict[str, float] = {}

    def boost_for(self, memory_id: str) -> float:
        """Return a clamped personal boost in ``[-1, 1]``.

        Falls back to ``0.0`` on any backend failure — the rerank loop
        must never raise because of personalisation.
        """
        if memory_id in self._cache:
            return self._cache[memory_id]
        try:
            value = float(
                self._store.personal_boost(memory_id, user_id=self._user_id)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("PersonalBoostProvider lookup failed: %s", exc)
            value = 0.0
        value = max(-1.0, min(1.0, value))
        self._cache[memory_id] = value
        return value

    def prime(self, memory_ids) -> None:  # pragma: no cover - convenience
        """Warm the cache for a batch of ids (future Phase F7 batching)."""
        for mid in memory_ids:
            self.boost_for(mid)


__all__ = ["PersonalBoostProvider"]
