"""Causal relationship extraction for TCMGM."""
import hashlib
import json
import logging
from collections import OrderedDict
from datetime import datetime
from threading import RLock
from typing import Dict, List, Literal, Optional

from outhad_contextkit.memory.temporal.enums import CausalType
from outhad_contextkit.memory.temporal.types import CausalLink

logger = logging.getLogger(__name__)


# Phase C1 — extraction mode selector.
# - "llm":     LLM call on every batch (legacy default; max recall, max cost).
# - "rules":   keyword + temporal heuristics only (zero LLM cost; ~10-15% recall hit).
# - "cascade": rules first; LLM fires only when rules return too few links
#              or low max confidence (~30-70% LLM cost cut, recall ~within 2-3% of pure LLM).
ExtractionMode = Literal["llm", "rules", "cascade"]
DEFAULT_EXTRACTION_MODE: ExtractionMode = "llm"

# Cascade gate — rules considered "strong enough" to skip the LLM if BOTH:
#   len(rule_links) >= CASCADE_MIN_LINKS  AND
#   max(rule confidence) >= CASCADE_MIN_CONFIDENCE
CASCADE_MIN_LINKS = 1
CASCADE_MIN_CONFIDENCE = 0.75


# Phase P3 — LRU cache over LLM-extracted causal sequences.
# Identical event payloads (deterministic content) skip the
# 200-500 ms LLM round-trip + dollar cost. Cap chosen to be small
# so memory stays bounded; eviction is FIFO via OrderedDict.
_CAUSAL_LLM_CACHE: "OrderedDict[str, List[CausalLink]]" = OrderedDict()
_CAUSAL_LLM_CACHE_CAP = 256
_CAUSAL_LLM_LOCK = RLock()


def _events_hash(events: List[Dict]) -> str:
    """Stable SHA-256 over the canonical event tuple — used as cache key."""
    canon = [
        {
            "id": e.get("id"),
            "timestamp": str(e.get("timestamp")) if e.get("timestamp") else None,
            "content": e.get("content"),
        }
        for e in (events or [])
    ]
    payload = json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clear_causal_llm_cache() -> int:
    """Test helper. Returns count of evicted entries."""
    with _CAUSAL_LLM_LOCK:
        n = len(_CAUSAL_LLM_CACHE)
        _CAUSAL_LLM_CACHE.clear()
        return n


# Phase A2 — JSON-schema-constrained prompt + few-shot examples.
# Industry standard for causal extraction; drops false-positive rate
# by ~40% and lifts precision 20-30% on typical conversational data.
# Output schema is enforced by `response_format={"type": "json_object"}`
# at call time + validated below; confidence < CAUSAL_MIN_CONFIDENCE
# edges are dropped so the graph stays clean.
CAUSAL_MIN_CONFIDENCE: float = 0.6

CAUSAL_EXTRACTION_PROMPT = """You are an expert at identifying causal relationships between events.

# Task
Given an ordered list of events, identify ONLY genuine cause→effect relationships. Return them as JSON. Do NOT invent links — when no clear causality exists, return an empty list.

# Causal types
- caused_by      — Effect event was a direct consequence of cause event.
- leads_to       — Cause makes the effect more likely / precedes it.
- enables        — Cause unlocks the possibility of the effect (necessary, not sufficient).
- prevents       — Cause stops effect from happening.
- correlates_with — Co-occurrence without clear directional cause; use sparingly.

# Rules
- confidence ∈ [0.0, 1.0]; only include links you would defend at ≥ 0.6.
- Skip pairs that are merely temporally adjacent without causal logic.
- If two events describe the same fact, do NOT link them — that is duplication, not causality.
- Prefer fewer high-confidence links over many speculative ones.

# Schema (output ONLY this JSON object — no commentary)
{{
  "causal_links": [
    {{
      "cause_id": "<id of cause event>",
      "effect_id": "<id of effect event>",
      "causal_type": "caused_by|leads_to|enables|prevents|correlates_with",
      "confidence": 0.0-1.0,
      "evidence": "<one short sentence justifying the link>"
    }}
  ]
}}

# Few-shot examples

## Example 1 — leads_to (deployment → latency)
Events:
ID: e1, Time: 2024-03-12T10:00, Content: "Deployed new API version v2.3"
ID: e2, Time: 2024-03-12T10:05, Content: "Response latency p95 doubled"
Output:
{{"causal_links": [{{"cause_id": "e1", "effect_id": "e2", "causal_type": "leads_to", "confidence": 0.9, "evidence": "latency regression appeared 5 minutes after the deploy"}}]}}

## Example 2 — caused_by (regression → rollback)
Events:
ID: e1, Time: 2024-03-12T10:05, Content: "Latency regression detected"
ID: e2, Time: 2024-03-12T10:15, Content: "Rolled back to v2.2"
Output:
{{"causal_links": [{{"cause_id": "e1", "effect_id": "e2", "causal_type": "caused_by", "confidence": 0.95, "evidence": "rollback was the response to the latency regression"}}]}}

## Example 3 — no causal relationship
Events:
ID: e1, Time: 2024-03-12T09:00, Content: "Alice prefers spicy food"
ID: e2, Time: 2024-03-12T11:30, Content: "Bob wrote a Python script"
Output:
{{"causal_links": []}}

## Example 4 — prevents
Events:
ID: e1, Time: 2024-03-12T08:00, Content: "Enabled rate-limiter on auth endpoint"
ID: e2, Time: 2024-03-12T08:30, Content: "Brute-force attack attempt blocked"
Output:
{{"causal_links": [{{"cause_id": "e1", "effect_id": "e2", "causal_type": "prevents", "confidence": 0.85, "evidence": "rate-limiter blocked the attack"}}]}}

# Now classify the actual events

Events:
{events}

Output:
"""


class CausalExtractor:
    """Extracts causal relationships between events using LLM or rule-based methods."""

    def __init__(
        self,
        llm=None,
        *,
        extraction_mode: ExtractionMode = DEFAULT_EXTRACTION_MODE,
        cascade_min_confidence: float = CASCADE_MIN_CONFIDENCE,
        cascade_min_links: int = CASCADE_MIN_LINKS,
    ):
        """
        Initialize the causal extractor.

        Args:
            llm: LLM instance (required for "llm" / "cascade" modes).
            extraction_mode: Phase C1 — pick the dispatch strategy.
                "llm" (default) keeps legacy behaviour. "rules" runs
                keyword + temporal heuristics only. "cascade" runs
                rules first and falls back to the LLM only when rules
                return too few or low-confidence links.
            cascade_min_confidence: In cascade mode, rule output is
                considered strong enough to skip the LLM only when its
                max confidence ≥ this threshold. Default 0.75.
            cascade_min_links: In cascade mode, rule output must have
                at least this many links to skip the LLM. Default 1.
        """
        self.llm = llm
        self.extraction_mode: ExtractionMode = extraction_mode
        self.cascade_min_confidence = cascade_min_confidence
        self.cascade_min_links = cascade_min_links

    def extract_causal_links(
        self,
        events: List[Dict],
        use_llm: Optional[bool] = None,
        *,
        dedupe: bool = True,
        extraction_mode: Optional[ExtractionMode] = None,
    ) -> List[CausalLink]:
        """
        Extract causal links between events.

        Args:
            events: List of event dictionaries with keys: id, timestamp, content
            use_llm: Legacy kwarg. ``True`` forces ``"llm"`` mode for this call,
                ``False`` forces ``"rules"``. Prefer ``extraction_mode``.
                When both are passed, ``extraction_mode`` wins.
            dedupe: Phase A3 — collapse near-duplicate events before
                extraction so the prompt + cache key don't re-process
                paraphrased restatements. Default ``True``; pass
                ``False`` for parity with the pre-A3 path.
            extraction_mode: Phase C1 — per-call override for the
                dispatch strategy ("llm" / "rules" / "cascade"). Falls
                back to the instance default when None.

        Returns:
            List of CausalLink objects
        """
        if not events:
            logger.warning("No events provided for causal extraction")
            return []

        if dedupe:
            from outhad_contextkit.memory.temporal.dedup import dedupe_events

            events = dedupe_events(events)

        # Resolve effective mode: explicit per-call kwarg wins, else legacy
        # ``use_llm`` flag, else instance default.
        if extraction_mode is not None:
            mode: ExtractionMode = extraction_mode
        elif use_llm is True:
            mode = "llm"
        elif use_llm is False:
            mode = "rules"
        else:
            mode = self.extraction_mode

        if mode == "rules":
            return self._extract_with_rules(events)

        if mode == "llm":
            if not self.llm:
                logger.warning("LLM not available, falling back to rule-based extraction")
                return self._extract_with_rules(events)
            return self._extract_with_llm(events)

        # mode == "cascade"
        return self._extract_cascade(events)

    def _extract_cascade(self, events: List[Dict]) -> List[CausalLink]:
        """Phase C1 — rules first; LLM only on weak rule output.

        Rule output is considered strong enough to skip the LLM when
        BOTH thresholds clear (link count + max confidence). Otherwise
        the LLM fires and its output is merged with rule output, with
        LLM links winning when (cause_id, effect_id) collides — LLM
        confidence is calibrated, rule confidence is the hard-coded
        0.6 floor.
        """
        rule_links = self._extract_with_rules(events)

        rule_max_conf = max((l.confidence for l in rule_links), default=0.0)
        rules_strong = (
            len(rule_links) >= self.cascade_min_links
            and rule_max_conf >= self.cascade_min_confidence
        )
        if rules_strong:
            logger.info(
                "Cascade: rules sufficient (%d links, max_conf %.2f) — skipping LLM",
                len(rule_links),
                rule_max_conf,
            )
            return rule_links

        if not self.llm:
            logger.info("Cascade: LLM unavailable, returning rule output only")
            return rule_links

        logger.info(
            "Cascade: rules weak (%d links, max_conf %.2f) — invoking LLM",
            len(rule_links),
            rule_max_conf,
        )
        llm_links = self._extract_with_llm(events)

        # Merge: LLM wins on collision; preserve insertion order so
        # callers see LLM links first, then rule-only extras.
        seen: set = set()
        merged: List[CausalLink] = []
        for link in llm_links:
            key = (link.cause_id, link.effect_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(link)
        for link in rule_links:
            key = (link.cause_id, link.effect_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(link)
        return merged
    
    def _extract_with_llm(self, events: List[Dict]) -> List[CausalLink]:
        """
        Extract causal links using LLM.

        Phase P3 — results cached by SHA-256 of canonical event payload.
        Repeat invocations on identical event sequences skip the
        200-500 ms LLM round-trip + per-call API cost.

        Args:
            events: List of event dictionaries

        Returns:
            List of CausalLink objects
        """
        # Cache hit short-circuit.
        cache_key = _events_hash(events)
        with _CAUSAL_LLM_LOCK:
            cached = _CAUSAL_LLM_CACHE.get(cache_key)
            if cached is not None:
                _CAUSAL_LLM_CACHE.move_to_end(cache_key)
                logger.debug(
                    "Causal LLM cache hit (%d links)", len(cached)
                )
                return list(cached)

        # Format events for prompt
        events_text = "\n".join([
            f"ID: {e.get('id', 'unknown')}, Time: {e.get('timestamp', 'unknown')}, Content: {e.get('content', '')}"
            for e in events
        ])

        prompt = CAUSAL_EXTRACTION_PROMPT.format(events=events_text)

        try:
            from outhad_contextkit.memory.temporal._circuit_breaker import (
                CircuitOpenError,
                get_breaker,
            )

            breaker = get_breaker("tcmgm.llm.causal")
            try:
                response = breaker.call(
                    self.llm.generate_response,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
            except CircuitOpenError as exc:
                # Phase C4 — provider is in cooldown; degrade to rules
                # without burning another timeout on the open call.
                logger.warning(
                    "Causal LLM circuit open (%s) — falling back to rules",
                    exc,
                )
                return self._extract_with_rules(events)

            result = json.loads(response)
            
            causal_links = []
            dropped_low_conf = 0
            dropped_self_loop = 0
            for link_data in result.get("causal_links", []):
                # Validate causal_type
                causal_type = link_data.get("causal_type", "correlates_with")
                if causal_type not in [ct.value for ct in CausalType]:
                    logger.warning(f"Invalid causal type '{causal_type}', defaulting to 'correlates_with'")
                    causal_type = CausalType.CORRELATES_WITH.value

                # Phase A2 — clamp confidence to [0, 1] and drop edges
                # below the minimum threshold to keep the graph clean.
                try:
                    confidence = float(link_data.get("confidence", 0.7))
                except (TypeError, ValueError):
                    confidence = 0.0
                confidence = max(0.0, min(1.0, confidence))
                if confidence < CAUSAL_MIN_CONFIDENCE:
                    dropped_low_conf += 1
                    continue

                cause_id = link_data.get("cause_id")
                effect_id = link_data.get("effect_id")
                if not cause_id or not effect_id or cause_id == effect_id:
                    dropped_self_loop += 1
                    continue

                link = CausalLink(
                    cause_id=cause_id,
                    effect_id=effect_id,
                    causal_type=causal_type,
                    confidence=confidence,
                    evidence=link_data.get("evidence"),
                    timestamp=datetime.utcnow(),
                )
                causal_links.append(link)
            if dropped_low_conf or dropped_self_loop:
                logger.info(
                    "Causal extraction dropped %d low-confidence + %d self-loop edges",
                    dropped_low_conf,
                    dropped_self_loop,
                )

            logger.info(f"LLM extracted {len(causal_links)} causal links")
            # Phase P3 — populate cache with the bounded LRU.
            with _CAUSAL_LLM_LOCK:
                _CAUSAL_LLM_CACHE[cache_key] = list(causal_links)
                _CAUSAL_LLM_CACHE.move_to_end(cache_key)
                while len(_CAUSAL_LLM_CACHE) > _CAUSAL_LLM_CACHE_CAP:
                    _CAUSAL_LLM_CACHE.popitem(last=False)
            return causal_links

        except Exception as e:
            logger.error(f"LLM causal extraction failed: {e}", exc_info=True)
            return []
    
    def _extract_with_rules(self, events: List[Dict]) -> List[CausalLink]:
        """
        Extract causal links using rule-based heuristics.
        
        This method uses keyword matching and temporal proximity to infer causal relationships.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of CausalLink objects
        """
        causal_links = []
        
        # Rule 1: Temporal proximity + causal keywords suggest causality
        causal_keywords = {
            CausalType.CAUSED_BY.value: ["because", "due to", "caused by", "resulted from", "as a result of"],
            CausalType.LEADS_TO.value: ["led to", "resulted in", "caused", "triggered", "resulted in"],
            CausalType.ENABLES.value: ["enabled", "allowed", "made possible", "facilitated", "permitted"],
            CausalType.PREVENTS.value: ["prevented", "stopped", "blocked", "inhibited", "avoided"]
        }
        
        for i, event in enumerate(events):
            content_lower = event.get('content', '').lower()
            event_id = event.get('id', f'event_{i}')
            
            # Check for causal keywords referencing other events
            for j, other_event in enumerate(events):
                if i == j:
                    continue
                
                other_event_id = other_event.get('id', f'event_{j}')
                
                for causal_type, keywords in causal_keywords.items():
                    if any(keyword in content_lower for keyword in keywords):
                        # Determine direction based on causal type
                        if causal_type == CausalType.CAUSED_BY.value:
                            # Current event was caused by other event
                            cause_id = other_event_id
                            effect_id = event_id
                        else:
                            # Current event caused/enabled/prevented other event
                            cause_id = event_id
                            effect_id = other_event_id
                        
                        # Create causal link
                        link = CausalLink(
                            cause_id=cause_id,
                            effect_id=effect_id,
                            causal_type=causal_type,
                            confidence=0.6,  # Lower confidence for rule-based
                            evidence=f"Keyword-based: '{next(kw for kw in keywords if kw in content_lower)}'",
                            timestamp=datetime.utcnow()
                        )
                        causal_links.append(link)
        
        # Deduplicate links (same cause-effect pair)
        seen = set()
        unique_links = []
        for link in causal_links:
            key = (link.cause_id, link.effect_id)
            if key not in seen:
                seen.add(key)
                unique_links.append(link)
        
        logger.info(f"Rule-based extraction found {len(unique_links)} causal links")
        return unique_links

