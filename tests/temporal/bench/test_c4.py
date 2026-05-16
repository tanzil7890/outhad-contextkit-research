"""provider circuit breaker.

Two layers covered:

* The standalone ``CircuitBreaker`` state machine (closed → open →
  half_open → closed | re-open) under a fake monotonic clock so tests
  run in milliseconds, not seconds.
* The wiring at the causal-LLM call site: when the breaker is open
  the extractor must skip the LLM call and degrade to the rule path
  without raising.
"""
from __future__ import annotations

import json
from typing import List

import pytest

from outhad_contextkit.memory.temporal._circuit_breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    CircuitOpenError,
    clear_breaker_registry,
    get_breaker,
    reset_breaker,
)


@pytest.fixture(autouse=True)
def _wipe_registry():
    clear_breaker_registry()
    yield
    clear_breaker_registry()


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_default_state_is_closed():
    b = CircuitBreaker("test")
    assert b.state == CLOSED
    assert b.consecutive_failures == 0


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        CircuitBreaker("test", failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker("test", recovery_timeout=-1)


def test_success_keeps_breaker_closed():
    b = CircuitBreaker("t", failure_threshold=2)
    out = b.call(lambda: "ok")
    assert out == "ok"
    assert b.state == CLOSED


def test_failures_below_threshold_keep_breaker_closed():
    b = CircuitBreaker("t", failure_threshold=3)

    def boom():
        raise RuntimeError("nope")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            b.call(boom)
    assert b.state == CLOSED
    assert b.consecutive_failures == 2


def test_threshold_failures_trip_breaker_open():
    b = CircuitBreaker("t", failure_threshold=3)

    def boom():
        raise RuntimeError("nope")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            b.call(boom)
    assert b.state == OPEN


def test_open_breaker_fast_fails_without_calling_fn():
    b = CircuitBreaker("t", failure_threshold=2)

    def boom():
        raise RuntimeError("nope")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            b.call(boom)

    calls = []

    def shouldnt_run():
        calls.append(1)

    with pytest.raises(CircuitOpenError):
        b.call(shouldnt_run)
    assert calls == []


def test_open_transitions_to_half_open_after_recovery_timeout():
    clock = _FakeClock(now=100.0)
    b = CircuitBreaker(
        "t", failure_threshold=1, recovery_timeout=10.0, clock=clock
    )

    with pytest.raises(RuntimeError):
        b.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert b.state == OPEN

    # Cooldown not elapsed — still open.
    clock.now = 105.0
    assert b.state == OPEN

    # Cooldown elapsed — moves to half_open on next state read.
    clock.now = 111.0
    assert b.state == HALF_OPEN


def test_half_open_success_closes_breaker():
    clock = _FakeClock(now=0.0)
    b = CircuitBreaker(
        "t", failure_threshold=1, recovery_timeout=5.0, clock=clock
    )

    with pytest.raises(RuntimeError):
        b.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    clock.now = 6.0  # past cooldown
    assert b.state == HALF_OPEN

    out = b.call(lambda: "recovered")
    assert out == "recovered"
    assert b.state == CLOSED
    assert b.consecutive_failures == 0


def test_half_open_failure_reopens_breaker():
    clock = _FakeClock(now=0.0)
    b = CircuitBreaker(
        "t", failure_threshold=1, recovery_timeout=5.0, clock=clock
    )

    with pytest.raises(RuntimeError):
        b.call(lambda: (_ for _ in ()).throw(RuntimeError("first")))
    clock.now = 6.0
    assert b.state == HALF_OPEN

    with pytest.raises(RuntimeError):
        b.call(lambda: (_ for _ in ()).throw(RuntimeError("probe-fail")))
    assert b.state == OPEN
    # Cooldown timer must restart.
    assert b._opened_at == 6.0


def test_manual_reset_returns_breaker_to_closed():
    b = CircuitBreaker("t", failure_threshold=1)
    with pytest.raises(RuntimeError):
        b.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert b.state == OPEN
    b.reset()
    assert b.state == CLOSED
    assert b.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_returns_same_instance_for_same_name():
    a = get_breaker("svc-a")
    b = get_breaker("svc-a")
    assert a is b


def test_registry_isolates_different_names():
    a = get_breaker("svc-a")
    b = get_breaker("svc-b")
    assert a is not b


def test_reset_breaker_returns_false_for_unknown_name():
    assert reset_breaker("nope") is False


# ---------------------------------------------------------------------------
# Causal-extractor wiring (C4 — degrade to rules when LLM circuit open)
# ---------------------------------------------------------------------------

class _FailingLLM:
    """LLM stub whose ``generate_response`` always raises."""

    def __init__(self):
        self.calls = 0

    def generate_response(self, messages, response_format):
        self.calls += 1
        raise RuntimeError("upstream down")


class _CountingLLM:
    """LLM stub that records call count and returns canned JSON."""

    def __init__(self, response):
        self.calls = 0
        self.response = response

    def generate_response(self, messages, response_format):
        self.calls += 1
        return json.dumps(self.response)


def _events_with_keyword():
    return [
        {"id": "e2", "timestamp": "2026-04-25T14:05", "content": "p95 spiked"},
        {
            "id": "e3",
            "timestamp": "2026-04-25T14:08",
            "content": "rolled back because spike",
        },
    ]


def test_open_breaker_short_circuits_llm_path_to_rules(monkeypatch):
    """After the breaker trips, the LLM is no longer called and rule output is returned."""
    from outhad_contextkit.memory.temporal._circuit_breaker import (
        CircuitBreaker,
    )
    from outhad_contextkit.memory.temporal.causal_extractor import (
        CausalExtractor,
        clear_causal_llm_cache,
    )

    clear_causal_llm_cache()

    # Drop the failure threshold so a single LLM error trips the breaker.
    breaker = CircuitBreaker("tcmgm.llm.causal", failure_threshold=1)
    monkeypatch.setattr(
        "outhad_contextkit.memory.temporal._circuit_breaker.get_breaker",
        lambda *args, **kwargs: breaker,
    )

    llm = _FailingLLM()
    ex = CausalExtractor(llm=llm, extraction_mode="llm")

    # First call: LLM raises → bubbles up but trips the breaker.
    out1 = ex.extract_causal_links(_events_with_keyword(), dedupe=False)
    # _extract_with_llm catches all exceptions and returns []; breaker now open.
    assert out1 == []
    assert llm.calls == 1
    assert breaker.state == OPEN

    # Second call: breaker open → LLM not invoked → rules path runs and
    # finds the "because" keyword.
    out2 = ex.extract_causal_links(
        _events_with_keyword(), dedupe=False
    )
    assert llm.calls == 1, "LLM must NOT be called when breaker is open"
    assert any(
        l.evidence and "because" in l.evidence for l in out2
    ), "expected rule fallback to surface keyword-matched link"

    clear_causal_llm_cache()


def test_closed_breaker_lets_llm_run_normally(monkeypatch):
    from outhad_contextkit.memory.temporal._circuit_breaker import (
        CircuitBreaker,
    )
    from outhad_contextkit.memory.temporal.causal_extractor import (
        CausalExtractor,
        clear_causal_llm_cache,
    )

    clear_causal_llm_cache()
    breaker = CircuitBreaker("tcmgm.llm.causal", failure_threshold=3)
    monkeypatch.setattr(
        "outhad_contextkit.memory.temporal._circuit_breaker.get_breaker",
        lambda *args, **kwargs: breaker,
    )

    llm = _CountingLLM({
        "causal_links": [
            {
                "cause_id": "e2",
                "effect_id": "e3",
                "causal_type": "caused_by",
                "confidence": 0.9,
                "evidence": "ok",
            }
        ]
    })
    ex = CausalExtractor(llm=llm, extraction_mode="llm")
    out = ex.extract_causal_links(_events_with_keyword(), dedupe=False)
    assert llm.calls == 1
    assert breaker.state == CLOSED
    assert any(l.cause_id == "e2" and l.effect_id == "e3" for l in out)
    clear_causal_llm_cache()
