"""cascade causal extraction.

Three modes are exercised:
* ``"llm"`` — legacy path; one LLM call per batch.
* ``"rules"`` — keyword + temporal heuristics; zero LLM calls.
* ``"cascade"`` — rules first; LLM fires only when rule output is too
  sparse or low-confidence to clear the gate.

The ``MockLLM`` records every call so we can assert exact call counts
under each mode + gate configuration.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from outhad_contextkit.memory.temporal.causal_extractor import (
    CASCADE_MIN_CONFIDENCE,
    CASCADE_MIN_LINKS,
    CausalExtractor,
    DEFAULT_EXTRACTION_MODE,
    clear_causal_llm_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockLLM:
    """Records prompts and replays a canned causal_links response."""

    def __init__(self, response_links: List[Dict[str, Any]]):
        self.response_links = response_links
        self.calls: List[str] = []

    def generate_response(
        self, messages: List[Dict[str, str]], response_format: Dict[str, str]
    ) -> str:
        self.calls.append(messages[0]["content"])
        return json.dumps({"causal_links": self.response_links})


def _events_with_keyword():
    return [
        {"id": "e1", "timestamp": "2026-04-25T14:00", "content": "deployed v2.3"},
        {"id": "e2", "timestamp": "2026-04-25T14:05", "content": "p95 spiked"},
        {
            "id": "e3",
            "timestamp": "2026-04-25T14:08",
            "content": "rolled back because spike",
        },
    ]


def _events_no_keyword():
    return [
        {"id": "e1", "timestamp": "2026-04-25T14:00", "content": "deployed v2.3"},
        {"id": "e2", "timestamp": "2026-04-25T14:05", "content": "p95 spiked"},
    ]


@pytest.fixture(autouse=True)
def _wipe_cache():
    clear_causal_llm_cache()
    yield
    clear_causal_llm_cache()


# ---------------------------------------------------------------------------
# Default + back-compat
# ---------------------------------------------------------------------------

def test_default_mode_constant():
    assert DEFAULT_EXTRACTION_MODE == "llm"
    assert CASCADE_MIN_LINKS >= 1
    assert 0.5 <= CASCADE_MIN_CONFIDENCE <= 1.0


def test_default_extractor_keeps_llm_mode():
    """Existing callers (no extraction_mode override) hit the LLM path."""
    llm = MockLLM([
        {"cause_id": "e1", "effect_id": "e2", "causal_type": "leads_to",
         "confidence": 0.9, "evidence": "x"},
    ])
    ex = CausalExtractor(llm=llm)
    out = ex.extract_causal_links(_events_with_keyword(), dedupe=False)
    assert len(llm.calls) == 1
    assert len(out) == 1


def test_use_llm_false_legacy_kwarg_routes_to_rules():
    """Back-compat: explicit use_llm=False still picks rule path."""
    llm = MockLLM([])
    ex = CausalExtractor(llm=llm)
    out = ex.extract_causal_links(_events_with_keyword(), use_llm=False, dedupe=False)
    assert len(llm.calls) == 0
    # rule path picks up "because" in e3
    assert any(l.evidence and "because" in l.evidence for l in out)


# ---------------------------------------------------------------------------
# Mode = "rules"
# ---------------------------------------------------------------------------

def test_rules_mode_skips_llm_entirely():
    llm = MockLLM([])
    ex = CausalExtractor(llm=llm, extraction_mode="rules")
    out = ex.extract_causal_links(_events_with_keyword(), dedupe=False)
    assert len(llm.calls) == 0
    # Rule keyword-match fires on "because"
    assert len(out) >= 1
    assert all(l.confidence == 0.6 for l in out)


def test_rules_mode_returns_empty_when_no_keywords():
    llm = MockLLM([])
    ex = CausalExtractor(llm=llm, extraction_mode="rules")
    out = ex.extract_causal_links(_events_no_keyword(), dedupe=False)
    assert len(llm.calls) == 0
    assert out == []


def test_per_call_extraction_mode_overrides_instance():
    llm = MockLLM([])
    ex = CausalExtractor(llm=llm, extraction_mode="llm")
    out = ex.extract_causal_links(
        _events_with_keyword(), extraction_mode="rules", dedupe=False
    )
    assert len(llm.calls) == 0
    assert len(out) >= 1


# ---------------------------------------------------------------------------
# Mode = "cascade"
# ---------------------------------------------------------------------------

def test_cascade_skips_llm_when_rules_strong():
    """Rules with conf >= cascade_min_confidence skip the LLM."""
    llm = MockLLM([
        {"cause_id": "e1", "effect_id": "e2", "causal_type": "leads_to",
         "confidence": 0.9, "evidence": "should-not-fire"},
    ])
    # Lower the threshold so the hard-coded 0.6 rule confidence clears the gate.
    ex = CausalExtractor(
        llm=llm,
        extraction_mode="cascade",
        cascade_min_confidence=0.5,
        cascade_min_links=1,
    )
    out = ex.extract_causal_links(_events_with_keyword(), dedupe=False)
    assert len(llm.calls) == 0
    assert len(out) >= 1


def test_cascade_invokes_llm_when_rules_empty():
    """No keyword in events → rules empty → LLM must fire."""
    llm = MockLLM([
        {"cause_id": "e1", "effect_id": "e2", "causal_type": "leads_to",
         "confidence": 0.92, "evidence": "temporal proximity"},
    ])
    ex = CausalExtractor(llm=llm, extraction_mode="cascade")
    out = ex.extract_causal_links(_events_no_keyword(), dedupe=False)
    assert len(llm.calls) == 1
    assert any(l.cause_id == "e1" and l.effect_id == "e2" for l in out)


def test_cascade_invokes_llm_when_rules_low_confidence():
    """Default cascade gate is 0.75; rule confidence 0.6 < 0.75 → LLM fires."""
    llm = MockLLM([
        {"cause_id": "e1", "effect_id": "e3", "causal_type": "leads_to",
         "confidence": 0.93, "evidence": "deploy preceded rollback"},
    ])
    ex = CausalExtractor(llm=llm, extraction_mode="cascade")
    out = ex.extract_causal_links(_events_with_keyword(), dedupe=False)
    assert len(llm.calls) == 1
    # Both LLM and rule output present in merged result.
    keys = {(l.cause_id, l.effect_id) for l in out}
    assert ("e1", "e3") in keys


def test_cascade_merge_llm_wins_on_collision():
    """Rule + LLM produce same (cause, effect) → LLM link kept (calibrated conf)."""
    # Rule path will produce e2 → e3 caused_by 'because' at conf 0.6.
    # LLM returns the same edge at conf 0.95 — LLM must win.
    llm = MockLLM([
        {"cause_id": "e2", "effect_id": "e3", "causal_type": "caused_by",
         "confidence": 0.95, "evidence": "explicit causation"},
    ])
    ex = CausalExtractor(
        llm=llm,
        extraction_mode="cascade",
        cascade_min_confidence=0.99,  # force LLM to fire
    )
    out = ex.extract_causal_links(_events_with_keyword(), dedupe=False)
    collided = [l for l in out if l.cause_id == "e2" and l.effect_id == "e3"]
    assert len(collided) == 1
    assert collided[0].confidence == pytest.approx(0.95)
    assert collided[0].evidence == "explicit causation"


def test_cascade_no_llm_available_returns_rule_output():
    """No LLM configured + rules empty + cascade → empty output, no crash."""
    ex = CausalExtractor(llm=None, extraction_mode="cascade")
    out = ex.extract_causal_links(_events_no_keyword(), dedupe=False)
    assert out == []


def test_cascade_no_llm_available_returns_rule_output_when_keyword_present():
    """No LLM + rules weak (below threshold) — return whatever rules produced."""
    ex = CausalExtractor(
        llm=None,
        extraction_mode="cascade",
        cascade_min_confidence=0.99,
    )
    out = ex.extract_causal_links(_events_with_keyword(), dedupe=False)
    assert len(out) >= 1


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

def test_empty_events_short_circuits_in_every_mode():
    llm = MockLLM([])
    for mode in ("llm", "rules", "cascade"):
        ex = CausalExtractor(llm=llm, extraction_mode=mode)
        assert ex.extract_causal_links([], dedupe=False) == []
    assert llm.calls == []
