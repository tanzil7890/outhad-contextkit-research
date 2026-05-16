"""provider circuit breaker.

External providers (LLM, Whisper, embedding APIs) fail in bursts: an
upstream incident, a quota cap, a regional brown-out. Naive retry
loops keep hammering a downed provider, multiplying latency and cost
on every ``add()`` while the user waits for each call to time out.

A circuit breaker shifts to "fast fail" mode once the failure rate
crosses a threshold — the next ``call()`` raises immediately, the
caller falls back to its degraded path (rule-based extraction, cached
embedding, placeholder vector), and we wait out the upstream incident
without burning resources.

Three states (industry-standard pattern):

* **closed** — normal operation. Every call goes through. Failures
  increment a counter; on ``failure_threshold`` consecutive failures
  the breaker trips to ``open``.
* **open** — fast-fail. ``call()`` raises ``CircuitOpenError`` without
  touching the wrapped function. After ``recovery_timeout`` seconds
  the breaker shifts to ``half_open``.
* **half_open** — probe. The next call is allowed through. Success →
  back to ``closed`` (counter reset). Failure → back to ``open``
  (cooldown restarts).

Thread-safe via a single ``RLock``. Caller registry is keyed by
breaker name, so different providers (``llm`` / ``whisper`` /
``openai-embed``) share one registry but maintain independent state.

Usage::

    from outhad_contextkit.memory.temporal._circuit_breaker import (
        CircuitBreaker, CircuitOpenError, get_breaker,
    )

    breaker = get_breaker("llm", failure_threshold=3, recovery_timeout=30.0)

    try:
        out = breaker.call(self.llm.generate_response, messages=...)
    except CircuitOpenError:
        return self._extract_with_rules(events)  # degraded path
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is short-circuited because the breaker is open."""


class CircuitBreaker:
    """Per-provider circuit breaker with closed → open → half-open transitions."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        Args:
            name: Identifier surfaced in logs and metrics
                (e.g. ``"llm"``, ``"whisper"``).
            failure_threshold: Consecutive failures that trip the
                breaker. Default 3.
            recovery_timeout: Seconds in ``open`` before the breaker
                accepts a probe call (transitions to ``half_open``).
                Default 30 s.
            clock: Test seam — defaults to ``time.monotonic``.
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout must be >= 0")
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._state = CLOSED
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.RLock()

    # ---- introspection -------------------------------------------------
    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    # ---- main entry ----------------------------------------------------
    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """Run ``fn`` if the breaker permits; raise ``CircuitOpenError`` otherwise.

        Bookkeeping:
          * closed + success → counter reset, stays closed
          * closed + failure → counter++; trips to open when threshold reached
          * open + (cooldown elapsed) → moves to half_open; one call admitted
          * open + (cooldown still active) → fast-fail
          * half_open + success → resets to closed
          * half_open + failure → back to open; cooldown restarts
        """
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == OPEN:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open — fast-failing "
                    f"({self._consecutive_failures} consecutive failures)"
                )

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result

    # ---- manual controls ----------------------------------------------
    def reset(self) -> None:
        """Force-reset to ``closed`` and clear failure counter."""
        with self._lock:
            self._state = CLOSED
            self._consecutive_failures = 0
            self._opened_at = None

    # ---- internals ----------------------------------------------------
    def _maybe_transition_to_half_open(self) -> None:
        if self._state != OPEN or self._opened_at is None:
            return
        if self._clock() - self._opened_at >= self.recovery_timeout:
            logger.info(
                "Circuit '%s' cooldown elapsed — transitioning to half_open",
                self.name,
            )
            self._state = HALF_OPEN

    def _record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._state == HALF_OPEN:
                logger.warning(
                    "Circuit '%s' probe failed — re-opening", self.name
                )
                self._state = OPEN
                self._opened_at = self._clock()
                return
            if (
                self._state == CLOSED
                and self._consecutive_failures >= self.failure_threshold
            ):
                logger.warning(
                    "Circuit '%s' tripped open after %d consecutive failures",
                    self.name,
                    self._consecutive_failures,
                )
                self._state = OPEN
                self._opened_at = self._clock()

    def _record_success(self) -> None:
        with self._lock:
            if self._state == HALF_OPEN:
                logger.info("Circuit '%s' probe succeeded — closing", self.name)
                self._state = CLOSED
            self._consecutive_failures = 0


# ---------------------------------------------------------------------------
# Per-name registry — callers reuse the same breaker across requests.
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = threading.RLock()


def get_breaker(
    name: str,
    *,
    failure_threshold: int = 3,
    recovery_timeout: float = 30.0,
) -> CircuitBreaker:
    """Return the cached breaker for ``name``, creating it on first touch.

    Threshold + timeout from the first call win — subsequent calls do
    not re-configure an existing breaker. Use ``reset_breaker(name)``
    or ``clear_breaker_registry()`` to reconfigure in tests.
    """
    cached = _REGISTRY.get(name)
    if cached is not None:
        return cached
    with _REGISTRY_LOCK:
        cached = _REGISTRY.get(name)
        if cached is not None:
            return cached
        breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        _REGISTRY[name] = breaker
        return breaker


def reset_breaker(name: str) -> bool:
    """Reset one breaker; returns True if it was registered."""
    with _REGISTRY_LOCK:
        b = _REGISTRY.get(name)
        if b is None:
            return False
        b.reset()
        return True


def clear_breaker_registry() -> int:
    """Test helper — drop every registered breaker. Returns count cleared."""
    with _REGISTRY_LOCK:
        n = len(_REGISTRY)
        _REGISTRY.clear()
        return n


__all__ = [
    "CLOSED",
    "OPEN",
    "HALF_OPEN",
    "CircuitBreaker",
    "CircuitOpenError",
    "clear_breaker_registry",
    "get_breaker",
    "reset_breaker",
]
