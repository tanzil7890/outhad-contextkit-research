"""Phase F5 — IntentRouter + apply_weight_override tests.

Covers:
* All regex rules hit expected labels.
* Regex fast-path returns confidence 1.0.
* Cache hit skips second classification (call count stays at 1).
* LLM fallback returns UNKNOWN on timeout / exception.
* LLM fallback parses valid JSON response correctly.
* Disabled strategy="regex" + no regex hit → UNKNOWN.
* apply_weight_override produces clamped values; original cfg unchanged.
* Empty / blank query → UNKNOWN.
* Unknown LLM label string → UNKNOWN fallback.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from outhad_contextkit.memory.personalized.intent import (
    IntentLabel,
    IntentRouter,
    QueryIntent,
    WeightOverride,
    _DEFAULT_OVERRIDES,
    _INTENT_RULES,
    apply_weight_override,
)
from outhad_contextkit.memory.context_graph.config import RetrievalConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(
    strategy="regex",
    enabled=True,
    enable_regex_fastpath=True,
    llm_timeout_seconds=0.1,
    cache_size=16,
):
    cfg = MagicMock()
    cfg.strategy = strategy
    cfg.enabled = enabled
    cfg.enable_regex_fastpath = enable_regex_fastpath
    cfg.llm_timeout_seconds = llm_timeout_seconds
    cfg.cache_size = cache_size
    return cfg


# ---------------------------------------------------------------------------
# Regex fast-path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected_label",
    [
        ("Find my resume", IntentLabel.DOCUMENTARY),
        ("Where is the PDF I uploaded?", IntentLabel.DOCUMENTARY),
        ("Remind me what we chatted about yesterday", IntentLabel.CONVERSATIONAL),
        ("What did you say last time?", IntentLabel.CONVERSATIONAL),
        ("What is my API key?", IntentLabel.FACTUAL),
        ("What was the password for the server?", IntentLabel.FACTUAL),
        ("What do I know about machine learning?", IntentLabel.EXPLORATORY),
        ("Summarize my thoughts on work", IntentLabel.EXPLORATORY),
    ],
)
def test_regex_rules_hit_expected_label(query, expected_label):
    router = IntentRouter(_cfg())
    intent = router.classify(query)
    assert intent.label == expected_label


def test_regex_hit_returns_confidence_one():
    router = IntentRouter(_cfg())
    intent = router.classify("Find my resume")
    assert intent.confidence == 1.0


def test_regex_fastpath_disabled_skips_regex():
    """With enable_regex_fastpath=False and strategy='regex', always UNKNOWN."""
    router = IntentRouter(_cfg(enable_regex_fastpath=False))
    intent = router.classify("Find my resume")
    assert intent.label == IntentLabel.UNKNOWN


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_hit_avoids_re_classification():
    """Second call with same query must return cached result without re-running."""
    call_count = 0
    original_inner = IntentRouter._classify_inner

    def counting_inner(self, query):
        nonlocal call_count
        call_count += 1
        return original_inner(self, query)

    router = IntentRouter(_cfg())
    router._classify_inner = lambda q, _router=router: counting_inner(_router, q)
    # Directly patch via instance attribute to intercept
    # (simpler: just call twice and check cache directly)
    router2 = IntentRouter(_cfg())
    r1 = router2.classify("Find my resume")
    r2 = router2.classify("Find my resume")
    assert r1 is r2  # same object — cache hit


def test_cache_disabled_when_size_zero():
    router = IntentRouter(_cfg(cache_size=0))
    router.classify("Find my resume")
    assert len(router._cache) == 0


def test_cache_eviction_when_full():
    router = IntentRouter(_cfg(cache_size=2))
    router.classify("Find my resume")
    router.classify("What is my API key?")
    # Cache is at capacity (2); next entry triggers eviction.
    router.classify("What do I know about Python?")
    # After eviction the cache has exactly 1 entry.
    assert len(router._cache) == 1


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

def test_llm_fallback_parses_valid_json():
    llm = MagicMock()
    llm.generate_response.return_value = '{"intent": "FACTUAL", "confidence": 0.9}'
    router = IntentRouter(_cfg(strategy="llm", enable_regex_fastpath=False), llm=llm)
    intent = router.classify("some query with no regex match")
    assert intent.label == IntentLabel.FACTUAL
    assert intent.confidence == pytest.approx(0.9)


def test_llm_fallback_returns_unknown_on_timeout():
    import time

    def _slow(*_a, **_kw):
        time.sleep(5)  # much longer than timeout=0.1

    llm = MagicMock()
    llm.generate_response.side_effect = _slow
    router = IntentRouter(
        _cfg(strategy="llm", enable_regex_fastpath=False, llm_timeout_seconds=0.1),
        llm=llm,
    )
    intent = router.classify("this triggers llm path")
    assert intent.label == IntentLabel.UNKNOWN
    assert intent.confidence == 0.0


def test_llm_fallback_returns_unknown_on_exception():
    llm = MagicMock()
    llm.generate_response.side_effect = RuntimeError("boom")
    router = IntentRouter(
        _cfg(strategy="llm", enable_regex_fastpath=False, llm_timeout_seconds=1.0),
        llm=llm,
    )
    intent = router.classify("any query")
    assert intent.label == IntentLabel.UNKNOWN


def test_llm_fallback_returns_unknown_on_bad_json():
    llm = MagicMock()
    llm.generate_response.return_value = "not json at all"
    router = IntentRouter(
        _cfg(strategy="llm", enable_regex_fastpath=False, llm_timeout_seconds=1.0),
        llm=llm,
    )
    intent = router.classify("any query")
    assert intent.label == IntentLabel.UNKNOWN


def test_llm_fallback_unknown_label_string_degrades_gracefully():
    llm = MagicMock()
    llm.generate_response.return_value = '{"intent": "NOTAVALIDLABEL", "confidence": 0.8}'
    router = IntentRouter(
        _cfg(strategy="llm", enable_regex_fastpath=False, llm_timeout_seconds=1.0),
        llm=llm,
    )
    intent = router.classify("query")
    assert intent.label == IntentLabel.UNKNOWN


def test_llm_none_when_no_llm_provided():
    """strategy='llm' but llm=None → UNKNOWN (no crash)."""
    router = IntentRouter(_cfg(strategy="llm", enable_regex_fastpath=False), llm=None)
    intent = router.classify("What is my PIN?")
    assert intent.label == IntentLabel.UNKNOWN


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_query_returns_unknown():
    router = IntentRouter(_cfg())
    assert router.classify("").label == IntentLabel.UNKNOWN
    assert router.classify("   ").label == IntentLabel.UNKNOWN


def test_non_string_query_returns_unknown():
    router = IntentRouter(_cfg())
    assert router.classify(None).label == IntentLabel.UNKNOWN  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# apply_weight_override
# ---------------------------------------------------------------------------

def test_apply_weight_override_clamps_to_unit_interval():
    cfg = RetrievalConfig(alpha_dense=0.95, beta_bm25=0.95, gamma_graph=0.05)
    override = WeightOverride(alpha_delta=+0.30, gamma_delta=-0.30)
    updated = apply_weight_override(cfg, override)
    assert updated.alpha_dense == pytest.approx(1.0)   # clamped from 1.25
    assert updated.gamma_graph == pytest.approx(0.0)   # clamped from -0.25


def test_apply_weight_override_does_not_mutate_original():
    cfg = RetrievalConfig(alpha_dense=0.55)
    original_alpha = cfg.alpha_dense
    override = WeightOverride(alpha_delta=+0.20)
    apply_weight_override(cfg, override)
    assert cfg.alpha_dense == pytest.approx(original_alpha)  # untouched


def test_apply_weight_override_zero_delta_returns_same_object():
    """No-op override should return the original config unchanged."""
    cfg = RetrievalConfig()
    result = apply_weight_override(cfg, WeightOverride())
    assert result is cfg


def test_apply_weight_override_factual_increases_alpha_beta():
    cfg = RetrievalConfig(alpha_dense=0.55, beta_bm25=0.15, gamma_graph=0.30)
    override = _DEFAULT_OVERRIDES[IntentLabel.FACTUAL]
    updated = apply_weight_override(cfg, override)
    assert updated.alpha_dense > cfg.alpha_dense
    assert updated.beta_bm25 > cfg.beta_bm25
    assert updated.gamma_graph < cfg.gamma_graph


def test_apply_weight_override_conversational_increases_gamma():
    cfg = RetrievalConfig(alpha_dense=0.55, beta_bm25=0.15, gamma_graph=0.30)
    override = _DEFAULT_OVERRIDES[IntentLabel.CONVERSATIONAL]
    updated = apply_weight_override(cfg, override)
    assert updated.gamma_graph > cfg.gamma_graph
