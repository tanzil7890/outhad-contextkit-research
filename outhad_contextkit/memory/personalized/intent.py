"""Phase F5 — Query intent classifier.

Two-stage pipeline:
1. Regex fast-path  — configurable rules ordered by specificity; first hit wins.
2. LLM fallback     — called only when strategy is "llm" or "hybrid" and no regex
                      hit was found.

Results are cached in a simple dict (max_size evict-all when full). Fail-soft:
any error returns ``UNKNOWN`` with confidence 0.0 so retrieval always continues.

Usage::

    router = IntentRouter(cfg=mspr_cfg.intent, llm=memory.llm)
    intent = router.classify("What was my API key?")
    eff_cfg = apply_weight_override(retrieval_cfg, intent.override)
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class IntentLabel(str, Enum):
    FACTUAL = "FACTUAL"
    CONVERSATIONAL = "CONVERSATIONAL"
    DOCUMENTARY = "DOCUMENTARY"
    EXPLORATORY = "EXPLORATORY"
    UNKNOWN = "UNKNOWN"


@dataclass
class WeightOverride:
    """Additive deltas applied to ``RetrievalConfig`` weights for a given intent.

    All deltas are clamped so the resulting weight stays in [0, 1].
    ``edge_type_priors`` is reserved for F8 graph traversal weighting.
    """

    alpha_delta: float = 0.0
    beta_delta: float = 0.0
    gamma_delta: float = 0.0
    delta_delta: float = 0.0
    edge_type_priors: Dict[str, float] = field(default_factory=dict)


@dataclass
class QueryIntent:
    label: IntentLabel
    confidence: float
    override: WeightOverride


_UNKNOWN_INTENT = QueryIntent(
    label=IntentLabel.UNKNOWN,
    confidence=0.0,
    override=WeightOverride(),
)


# ---------------------------------------------------------------------------
# Regex fast-path rules — ordered: first match wins.
# ---------------------------------------------------------------------------

_INTENT_RULES: List[Tuple[IntentLabel, re.Pattern]] = [
    (
        IntentLabel.DOCUMENTARY,
        re.compile(
            r"\b(doc|document|resume|pdf|file|attachment|upload|link|contract|spreadsheet)\b",
            re.I,
        ),
    ),
    (
        IntentLabel.CONVERSATIONAL,
        re.compile(
            r"\b(remember|recall|chatted|said|told|last time|yesterday|conversation|discussed|talked)\b",
            re.I,
        ),
    ),
    (
        IntentLabel.FACTUAL,
        re.compile(
            r"^(what is|what'?s|when|where|how many|what was|who is|who'?s|pin|password|code|api key)\b",
            re.I,
        ),
    ),
    (
        IntentLabel.EXPLORATORY,
        re.compile(
            r"\b(think about|thoughts on|summarize|overview|tell me about|what do i know)\b",
            re.I,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Default weight overrides per intent label.
# ---------------------------------------------------------------------------

_DEFAULT_OVERRIDES: Dict[IntentLabel, WeightOverride] = {
    IntentLabel.FACTUAL: WeightOverride(
        alpha_delta=+0.10,
        beta_delta=+0.10,
        gamma_delta=-0.10,
    ),
    IntentLabel.CONVERSATIONAL: WeightOverride(
        gamma_delta=+0.15,
        edge_type_priors={"REPLY_TO": 1.25, "TEMPORAL_NEXT": 1.25},
    ),
    IntentLabel.DOCUMENTARY: WeightOverride(
        gamma_delta=+0.10,
        edge_type_priors={"DOCUMENT_LINK": 1.50},
    ),
    IntentLabel.EXPLORATORY: WeightOverride(),
    IntentLabel.UNKNOWN: WeightOverride(),
}


# ---------------------------------------------------------------------------
# Weight application — pure function, original config is never mutated.
# ---------------------------------------------------------------------------

def apply_weight_override(retrieval_cfg: Any, override: WeightOverride) -> Any:
    """Return a shallow copy of *retrieval_cfg* with intent deltas applied.

    Supports Pydantic v2 ``BaseModel`` (via ``model_copy``) and plain objects.
    All resulting weights are clamped to [0, 1].
    """

    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    field_map = [
        ("alpha_dense", override.alpha_delta),
        ("beta_bm25", override.beta_delta),
        ("gamma_graph", override.gamma_delta),
        ("delta_personal", override.delta_delta),
    ]
    updates = {
        attr: _clamp(getattr(retrieval_cfg, attr) + delta)
        for attr, delta in field_map
        if delta != 0.0 and hasattr(retrieval_cfg, attr)
    }
    if not updates:
        return retrieval_cfg  # nothing to change — return as-is (no copy)

    if hasattr(retrieval_cfg, "model_copy"):
        # Pydantic v2
        return retrieval_cfg.model_copy(update=updates)

    # Fallback for non-Pydantic objects (tests, mocks, etc.)
    import copy
    updated = copy.copy(retrieval_cfg)
    for k, v in updates.items():
        try:
            setattr(updated, k, v)
        except (AttributeError, TypeError):
            pass
    return updated


# ---------------------------------------------------------------------------
# IntentRouter
# ---------------------------------------------------------------------------

class IntentRouter:
    """Classify a query string into a ``QueryIntent``.

    Args:
        cfg: ``IntentConfig`` from ``MSPRConfig.intent``.
        llm: Memory LLM instance exposing ``generate_response(messages, ...)``.
             ``None`` disables LLM fallback.
        overrides: Mapping from ``IntentLabel`` to ``WeightOverride``.
                   Defaults to ``_DEFAULT_OVERRIDES``.
    """

    def __init__(
        self,
        cfg: Any,
        llm: Optional[Any] = None,
        overrides: Optional[Dict[IntentLabel, WeightOverride]] = None,
    ) -> None:
        self._cfg = cfg
        self._llm = llm
        self._overrides: Dict[IntentLabel, WeightOverride] = (
            overrides if overrides is not None else _DEFAULT_OVERRIDES
        )
        self._cache_size = max(0, int(getattr(cfg, "cache_size", 1024)))
        self._cache: Dict[str, QueryIntent] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> QueryIntent:
        """Classify *query*. Always returns a ``QueryIntent`` — never raises."""
        if not isinstance(query, str) or not query.strip():
            return _UNKNOWN_INTENT
        key = query.strip()
        if key in self._cache:
            return self._cache[key]
        try:
            result = self._classify_inner(key)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("IntentRouter.classify error: %s", exc)
            result = _UNKNOWN_INTENT
        self._store_cache(key, result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _store_cache(self, key: str, value: QueryIntent) -> None:
        if self._cache_size <= 0:
            return
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = value

    def _classify_inner(self, query: str) -> QueryIntent:
        # Stage 1 — regex fast-path
        if getattr(self._cfg, "enable_regex_fastpath", True):
            for label, pattern in _INTENT_RULES:
                if pattern.search(query):
                    return QueryIntent(
                        label=label,
                        confidence=1.0,
                        override=self._overrides.get(label, WeightOverride()),
                    )

        strategy = getattr(self._cfg, "strategy", "regex")

        # Stage 2 — LLM fallback
        if strategy in ("llm", "hybrid") and self._llm is not None:
            label, confidence = self._llm_classify(query)
            return QueryIntent(
                label=label,
                confidence=confidence,
                override=self._overrides.get(label, WeightOverride()),
            )

        return QueryIntent(
            label=IntentLabel.UNKNOWN,
            confidence=0.0,
            override=self._overrides.get(IntentLabel.UNKNOWN, WeightOverride()),
        )

    def _llm_classify(self, query: str) -> Tuple[IntentLabel, float]:
        from outhad_contextkit.memory.personalized.prompts import (
            INTENT_SYSTEM_PROMPT,
            INTENT_USER_TEMPLATE,
        )

        timeout = float(getattr(self._cfg, "llm_timeout_seconds", 2.0))
        result_holder: List[Optional[str]] = [None]
        exc_holder: List[Optional[Exception]] = [None]

        def _call() -> None:
            try:
                result_holder[0] = self._llm.generate_response(
                    messages=[
                        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": INTENT_USER_TEMPLATE.format(query=query),
                        },
                    ],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                exc_holder[0] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            logger.debug("LLM intent classification timed out after %.1fs", timeout)
            return IntentLabel.UNKNOWN, 0.0

        if exc_holder[0] is not None:
            logger.debug("LLM intent classification error: %s", exc_holder[0])
            return IntentLabel.UNKNOWN, 0.0

        response = result_holder[0]
        if not response:
            return IntentLabel.UNKNOWN, 0.0

        try:
            parsed = json.loads(response)
            label_str = str(parsed.get("intent", "UNKNOWN")).upper().strip()
            try:
                label = IntentLabel(label_str)
            except ValueError:
                label = IntentLabel.UNKNOWN
            confidence = float(max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))))
            return label, confidence
        except Exception as exc:
            logger.debug("LLM intent JSON parse error: %s", exc)
            return IntentLabel.UNKNOWN, 0.0
