""" ``PersonalizedRetrievalPipeline`` orchestrator.

The pipeline is a thin wrapper around the wiring already performed
inside :meth:`Memory._graph_first_rerank` (which by F7 covers intent
re-weighting, role hard/soft filtering, ε·frequency, δ·personal, and
ζ·success). It exists for two reasons:

1. **External callers** — applications that want to run the full MSPR
   stack without the rest of ``Memory.search()`` (e.g. a custom
   pre-fetched candidate list from a third-party recall step).
2. **Telemetry surface** — the pipeline is the natural place to emit
   the ``outhad_contextkit.mspr.search`` event with stage-by-stage
   metrics (intent label, role-filter count, latency).

The pipeline is constructed lazily by ``Memory._init_mspr`` only when
``mspr.enabled=True`` and stored on ``Memory._mspr_pipeline``. When
disabled, the pipeline reference stays ``None`` and ``Memory.search``
takes the pre-MSPR path.

The class is deliberately lightweight: it does not own state beyond a
back-reference to ``Memory``. All long-lived stores (FeedbackStore,
QuerySuccessStore, IntentRouter, RolePolicy) live on ``Memory``.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from outhad_contextkit.memory.telemetry import capture_event

logger = logging.getLogger(__name__)


class PersonalizedRetrievalPipeline:
    """High-level orchestrator for MSPR-enabled retrieval.

    Args:
        memory: A ``Memory`` instance. The pipeline reads its config
                and the lazily-initialised stores (``_feedback_store``,
                ``_success_store``, ``_intent_router``, ``_role_policy``).
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        seed_results: List[Dict[str, Any]],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_archived: bool = False,
    ) -> Dict[str, Any]:
        """Run the full MSPR retrieval flow on ``seed_results``.

        Returns the same shape as ``GraphFirstRetriever.retrieve`` —
        ``{"results": [...], "subgraph": [...]}`` — plus an ``mspr``
        sidecar dict with the stage-level telemetry that
        :meth:`Memory.search` also surfaces.

        The pipeline assumes the caller has already retrieved a seed
        set from the vector store / graph index. It does not run the
        recall step itself; that stays the caller's responsibility so
        the pipeline can be unit-tested in isolation.
        """
        m = self._memory
        if not getattr(getattr(m, "config", None), "mspr", None) or not (
            m.config.mspr.enabled
        ):
            return {"results": list(seed_results), "subgraph": []}

        role_ctx = m._build_role_ctx(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            filters=filters,
        )

        started = time.perf_counter()
        try:
            expanded = m._graph_first_rerank(
                query=query,
                seed_results=seed_results,
                limit=limit,
                include_archived=include_archived,
                role_ctx=role_ctx,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("MSPR pipeline rerank failed: %s", exc)
            expanded = None
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if expanded is None:
            return {"results": list(seed_results), "subgraph": []}

        # Emit the stage telemetry (no-op when telemetry is disabled
        # globally; PPMF sanitiser still applies to free-text fields).
        try:
            self._emit_telemetry(
                expanded,
                query=query,
                seed_count=len(seed_results),
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:  # pragma: no cover - telemetry never fatal
            logger.debug("MSPR telemetry emit failed: %s", exc)

        expanded = dict(expanded)
        expanded["mspr"] = {
            "latency_ms": elapsed_ms,
            "seed_count": len(seed_results),
            "result_count": len(expanded.get("results") or []),
        }
        return expanded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit_telemetry(
        self,
        expanded: Dict[str, Any],
        *,
        query: str,
        seed_count: int,
        elapsed_ms: int,
    ) -> None:
        """Fire ``outhad_contextkit.mspr.search`` (best-effort)."""
        results = expanded.get("results") or []
        ctx_payloads = [
            r.get("context_graph") or {} for r in results
        ]
        feedback_hits = sum(
            1 for c in ctx_payloads if abs(float(c.get("personal", 0.0))) > 0
        )
        success_hits = sum(
            1 for c in ctx_payloads if float(c.get("success", 0.0)) > 0
        )
        frequency_hits = sum(
            1 for c in ctx_payloads if float(c.get("frequency", 0.0)) > 0
        )

        cfg = self._memory.config.mspr
        capture_event(
            "outhad_contextkit.mspr.search",
            self._memory,
            {
                "intent_enabled": bool(cfg.intent.enabled),
                "feedback_enabled": bool(cfg.feedback.enabled),
                "frequency_enabled": bool(cfg.frequency.enabled),
                "role_enabled": bool(cfg.role.enabled),
                "success_enabled": bool(cfg.success.enabled),
                "seed_count": int(seed_count),
                "result_count": len(results),
                "feedback_hits": int(feedback_hits),
                "success_hits": int(success_hits),
                "frequency_hits": int(frequency_hits),
                "latency_ms": int(elapsed_ms),
                "sync_type": "sync",
            },
        )


__all__ = ["PersonalizedRetrievalPipeline"]
