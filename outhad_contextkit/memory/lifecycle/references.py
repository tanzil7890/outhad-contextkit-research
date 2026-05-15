"""Phase D6 — reference-based decay refresh.

Distinguishes "this memory was retrieved" (which already happens in
``Memory.search`` and bumps ``access_count`` on the returned top-K)
from "this memory was *used* by the agent in its response". Only the
latter should refresh freshness aggressively.

External code calls :meth:`Memory.record_reference(memory_id, ...)`
when the agent has injected the memory into a prompt; the tracker
bumps ``access_count`` + ``last_accessed_at`` and emits a
``node_referenced`` change-bus event so subscribers can drive
analytics dashboards without polling.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ReferenceTracker:
    """Stateless wrapper around the CGL ``record_access`` primitive.

    The tracker exists as its own class (instead of inlining the call
    on ``Memory``) so future weighting / batching can land here without
    touching the ``Memory`` API.
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    def record_reference(
        self,
        memory_id: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        strength: float = 1.0,
    ) -> Optional[int]:
        """Bump ``access_count`` + refresh ``last_accessed_at``.

        Returns the new access count, or ``None`` when the CGL is
        disabled or the memory is unknown. ``strength`` is reserved
        for future weighting; today the value is rounded down to the
        nearest non-negative integer delta (default 1).
        """
        cg = getattr(self._memory, "_context_graph", None)
        if cg is None:
            return None
        delta = max(0, int(round(float(strength))))
        if delta == 0:
            return None
        new_count: Optional[int] = None
        try:
            new_count = cg.record_access(
                memory_id, delta=delta, user_id=user_id
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("ReferenceTracker.record_access failed: %s", exc)
        # Emit a dedicated change-bus event so dashboards can
        # distinguish references from plain accesses.
        try:
            cg._emit(
                "node_referenced",
                memory_id,
                user_id,
                {
                    "agent_id": agent_id,
                    "run_id": run_id,
                    "strength": float(strength),
                    "access_count": new_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("ReferenceTracker emit failed: %s", exc)

        # Telemetry — best-effort.
        try:
            from outhad_contextkit.memory.telemetry import capture_event

            capture_event(
                "outhad_contextkit.lifecycle.reference",
                self._memory,
                {
                    "memory_id": memory_id,
                    "strength": float(strength),
                    "sync_type": "sync",
                },
            )
        except Exception:  # pragma: no cover - telemetry never fatal
            pass
        return new_count


__all__ = ["ReferenceTracker"]
