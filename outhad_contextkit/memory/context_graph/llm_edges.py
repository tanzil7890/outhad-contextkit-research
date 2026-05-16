""" LLM-driven semantic edge synthesis.

The ``LLMEdgeSynthesizer`` runs **after** structural topic edges have
been synthesised for a newly-inserted memory. It takes the top-K
topic-similar neighbours, asks an LLM to classify the relationship of
each pair (``SUPPORTS`` / ``CONTRADICTS`` / ``REFINES`` / ``ELABORATES``
/ ``NONE``), and returns a list of inferred edges which the builder
then persists via the ``ContextGraph`` facade.

Design goals:

* **Fail-soft.** LLM errors / timeouts / schema violations never break
  ``Memory.add``. On error, the synthesiser returns ``[]``.
* **Budgeted.** Every call is pre-checked against :class:`LLMBudget`.
  Exhaustion flips the user into a cooldown and short-circuits.
* **Deterministic JSON contract.** The prompt asks for strict JSON and
  the parser tolerates common LLM formatting quirks (fenced code blocks,
  trailing commentary) via a conservative JSON-object extractor.
* **Dependency-free.** No new optional extras — reuses ``Memory.llm``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

from outhad_contextkit.memory.context_graph.budget import LLMBudget
from outhad_contextkit.memory.context_graph.config import LLMEdgeConfig
from outhad_contextkit.memory.context_graph.types import EdgeType

logger = logging.getLogger(__name__)


LABEL_TO_EDGE_TYPE = {
    "SUPPORTS": EdgeType.SUPPORTS,
    "CONTRADICTS": EdgeType.CONTRADICTS,
    "REFINES": EdgeType.REFINES,
    "ELABORATES": EdgeType.ELABORATES,
}


SYSTEM_PROMPT = (
    "You classify the relationship between two user memories.\n"
    "Return ONLY a JSON object with exactly these keys: "
    '{"label": <one of SUPPORTS|CONTRADICTS|REFINES|ELABORATES|NONE>, '
    '"confidence": <float between 0 and 1>, '
    '"evidence": <short natural-language justification, ≤ 140 chars>}.\n'
    "Use NONE when neither memory clearly relates to the other. Do not "
    "invent facts; if unsure, return NONE with low confidence."
)

USER_PROMPT_TEMPLATE = 'A: "{text_a}"\nB: "{text_b}"'


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass
class InferredEdge:
    """An LLM-classified semantic edge awaiting persistence."""

    src: str
    dst: str
    type: EdgeType
    weight: float
    evidence: str
    confidence: float


class LLMResponseError(RuntimeError):
    """Raised when the LLM response cannot be parsed after retry."""


class LLMEdgeSynthesizer:
    """Classify candidate memory pairs into semantic CGL edges."""

    def __init__(
        self,
        llm: Any,
        cfg: LLMEdgeConfig,
        budget: LLMBudget,
        *,
        timer: Optional[Callable[[], float]] = None,
    ) -> None:
        self.llm = llm
        self.cfg = cfg
        self.budget = budget
        self._timer = timer

    # ---- public entry point -------------------------------------------
    def infer_edges(
        self,
        source_id: str,
        source_text: str,
        candidates: Sequence[Tuple[str, str, float]],
        *,
        user_id: Optional[str] = None,
    ) -> List[InferredEdge]:
        """Return up to ``cfg.max_pairs_per_insert`` inferred edges.

        ``candidates`` must be pre-filtered for the memory itself; each
        entry is ``(candidate_id, candidate_text, topic_similarity)``.
        """
        if not self.cfg.enabled or self.llm is None or not candidates:
            return []
        # Cheap structural filter: drop weak topic pairs before spending
        # any LLM tokens, then cap at ``max_pairs_per_insert``.
        filtered = [
            (cid, text, sim)
            for (cid, text, sim) in candidates
            if sim >= self.cfg.min_topic_similarity and cid != source_id
        ]
        filtered.sort(key=lambda item: item[2], reverse=True)
        filtered = filtered[: max(1, self.cfg.max_pairs_per_insert)]
        if not filtered:
            return []

        edges: List[InferredEdge] = []
        for cid, text, _sim in filtered:
            if not self.budget.has_capacity(
                user_id, self.cfg.tokens_per_call_estimate
            ):
                logger.warning(
                    "CGL LLM edge synthesis skipped: user=%s out of daily budget",
                    user_id,
                )
                break
            parsed = self._classify_pair(source_text, text)
            # Always charge the estimated cost so repeated flaky calls
            # still drain the budget and trip cooldown.
            self.budget.record_usage(user_id, self.cfg.tokens_per_call_estimate)
            if parsed is None:
                continue
            label, confidence, evidence = parsed
            if label == "NONE":
                continue
            if confidence < self.cfg.min_confidence:
                continue
            edge_type = LABEL_TO_EDGE_TYPE.get(label)
            if edge_type is None:
                continue
            edges.append(
                InferredEdge(
                    src=source_id,
                    dst=cid,
                    type=edge_type,
                    weight=float(confidence),
                    evidence=evidence[:280],
                    confidence=float(confidence),
                )
            )
        return edges

    # ---- helpers -------------------------------------------------------
    def _classify_pair(
        self,
        text_a: str,
        text_b: str,
    ) -> Optional[Tuple[str, float, str]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    text_a=(text_a or "").strip(),
                    text_b=(text_b or "").strip(),
                ),
            },
        ]
        attempts = 2 if self.cfg.retry_on_schema_violation else 1
        last_err: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                raw = self._invoke_llm(messages)
            except Exception as exc:  # noqa: BLE001 - fail-soft by design
                last_err = exc
                logger.debug(
                    "CGL LLM edge call failed (attempt %d/%d): %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                continue
            parsed = self._parse_response(raw)
            if parsed is not None:
                return parsed
            last_err = LLMResponseError(f"schema violation: {raw!r}")
            # Augment messages for retry with a clarifying hint.
            messages = messages + [
                {"role": "assistant", "content": raw or ""},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON. Reply with "
                        'ONLY {"label": ..., "confidence": ..., "evidence": ...}.'
                    ),
                },
            ]
        if last_err is not None:
            logger.debug("CGL LLM edge classification gave up: %s", last_err)
        return None

    def _invoke_llm(self, messages: List[dict]) -> str:
        # The LLMBase contract is ``generate_response(messages=...)``.
        # Callers pass anything that implements that. We do not attempt
        # JSON mode flags here because not every provider supports them;
        # the parser below tolerates wrapped payloads.
        if hasattr(self.llm, "generate_response"):
            return self.llm.generate_response(messages=messages) or ""
        if callable(self.llm):
            return self.llm(messages) or ""
        raise TypeError(
            f"LLMEdgeSynthesizer requires generate_response(messages) or callable LLM, "
            f"got {type(self.llm).__name__}"
        )

    @staticmethod
    def _parse_response(raw: Any) -> Optional[Tuple[str, float, str]]:
        if raw is None:
            return None
        text = raw if isinstance(raw, str) else str(raw)
        text = text.strip()
        if not text:
            return None
        # Tolerate fenced code blocks like ```json {...} ```.
        if text.startswith("```"):
            text = text.strip("`")
            text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE).strip()
        payload: Optional[dict] = None
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                payload = loaded
        except (json.JSONDecodeError, TypeError):
            match = _JSON_OBJECT_RE.search(text)
            if match is not None:
                try:
                    loaded = json.loads(match.group(0))
                    if isinstance(loaded, dict):
                        payload = loaded
                except (json.JSONDecodeError, TypeError):
                    payload = None
        if payload is None:
            return None
        label_raw = payload.get("label")
        if not isinstance(label_raw, str):
            return None
        label = label_raw.strip().upper()
        if label not in {"SUPPORTS", "CONTRADICTS", "REFINES", "ELABORATES", "NONE"}:
            return None
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        evidence = payload.get("evidence") or ""
        if not isinstance(evidence, str):
            evidence = str(evidence)
        return label, confidence, evidence


__all__ = ["InferredEdge", "LLMEdgeSynthesizer", "LLMResponseError"]
