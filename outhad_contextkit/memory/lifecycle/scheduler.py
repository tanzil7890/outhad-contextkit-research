"""periodic decay scheduler.

Runs ``ContextGraph.tick_decay()`` + ``Memory.archive_low()`` on a
daemon thread at ``interval_seconds`` cadence with optional jitter. Per
F8's pattern, telemetry events fire once per tick. Crashes inside
``_tick`` log + continue so a transient backend failure can never break
the loop.

Async callers (``AsyncMemory``) get an asyncio-task variant that does
not auto-start; they invoke ``await memory.start_decay_scheduler()``
when their event loop is alive.
"""
from __future__ import annotations

import asyncio
import logging
import random
import threading
from typing import Any, Optional

from outhad_contextkit.memory.lifecycle.config import ScheduleConfig

logger = logging.getLogger(__name__)


class DecayScheduler:
    """Daemon-thread variant for synchronous ``Memory``."""

    def __init__(self, memory: Any, *, cfg: ScheduleConfig) -> None:
        self._memory = memory
        self._cfg = cfg
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def start(self) -> None:
        """Start the daemon thread. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="MemoryDecayScheduler",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "DecayScheduler started (interval=%.1fs, jitter=±%.1fs)",
                self._cfg.interval_seconds,
                self._cfg.jitter_seconds,
            )

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal the loop to exit and join the thread (best-effort)."""
        with self._lock:
            if self._thread is None:
                return
            self._stop.set()
            t = self._thread
            self._thread = None
        t.join(timeout=max(0.0, float(timeout)))

    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def tick_now(self) -> None:
        """Run one tick synchronously (used by tests)."""
        self._tick()

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("DecayScheduler tick failed: %s", exc)
            jitter = self._cfg.jitter_seconds
            delta = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
            wait = max(1.0, self._cfg.interval_seconds + delta)
            self._stop.wait(timeout=wait)

    def _tick(self) -> None:
        cg = getattr(self._memory, "_context_graph", None)
        if cg is not None:
            try:
                cg.tick_decay()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("tick_decay failed: %s", exc)
        # Archive low-decay nodes when the helper is wired.
        archive_low = getattr(self._memory, "archive_low", None)
        if callable(archive_low):
            try:
                archive_low()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("archive_low failed: %s", exc)
        # Prune over-long version chains (D4).
        prune = getattr(self._memory, "_prune_version_chains", None)
        if callable(prune):
            try:
                prune()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("_prune_version_chains failed: %s", exc)
        # Telemetry — fail-soft; pulled from telemetry.capture_event so
        # it sanitises through PPMF when enabled.
        try:
            from outhad_contextkit.memory.telemetry import capture_event

            capture_event(
                "outhad_contextkit.lifecycle.tick",
                self._memory,
                {"sync_type": "sync"},
            )
        except Exception:  # pragma: no cover - telemetry never fatal
            pass


class AsyncDecayScheduler:
    """Asyncio-task variant for ``AsyncMemory``."""

    def __init__(self, memory: Any, *, cfg: ScheduleConfig) -> None:
        self._memory = memory
        self._cfg = cfg
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._loop(), name="AsyncMemoryDecayScheduler"
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None

    def is_running(self) -> bool:
        return bool(self._task is not None and not self._task.done())

    async def _loop(self) -> None:
        assert self._stop_event is not None
        # Sync scheduler shares the tick body; reuse it via a thin shim.
        sync = DecayScheduler(self._memory, cfg=self._cfg)
        while not self._stop_event.is_set():
            try:
                await asyncio.to_thread(sync._tick)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("AsyncDecayScheduler tick failed: %s", exc)
            jitter = self._cfg.jitter_seconds
            delta = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
            wait = max(1.0, self._cfg.interval_seconds + delta)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=wait
                )
            except asyncio.TimeoutError:
                continue


__all__ = ["DecayScheduler", "AsyncDecayScheduler"]
