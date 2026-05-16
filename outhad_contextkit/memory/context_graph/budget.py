""" Per-user LLM token / call budget for edge synthesis.

Keeps the CGL's LLM spend bounded without requiring external infra.
An in-memory map tracks tokens spent per ``(user_id, utc_date)`` and an
optional SQLite file can be passed in for cross-process / cross-run
durability.

The budget is a **pre-check**. ``has_capacity`` returns False once a
user is out for the day; ``record_usage`` is called after a successful
LLM call. If a call raises or times out, callers are expected to record
no usage (fail-soft). Budget exhaustion also triggers a short-lived
``cooldown`` to stop the hot loop from hammering the LLM on every
insert.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import sqlite3
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _utc_date() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d")


def _monotonic() -> float:
    # Separate helper to make the class trivially mockable in tests.
    import time

    return time.monotonic()


class LLMBudget:
    """Tracks daily token usage per-user, with optional SQLite durability."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS context_graph_llm_budget (
        user_id   TEXT NOT NULL,
        day       TEXT NOT NULL,
        tokens    INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, day)
    );
    """

    def __init__(
        self,
        daily_token_budget: int,
        *,
        cooldown_seconds: int = 900,
        persist_path: Optional[str] = None,
    ) -> None:
        self._budget = max(0, int(daily_token_budget))
        self._cooldown_seconds = max(0, int(cooldown_seconds))
        self._lock = threading.RLock()
        self._spent: Dict[Tuple[str, str], int] = {}
        self._cooldown_until: Dict[str, float] = {}
        self._conn: Optional[sqlite3.Connection] = None
        if persist_path:
            parent = os.path.dirname(os.path.abspath(persist_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._conn = sqlite3.connect(
                persist_path, check_same_thread=False, isolation_level=None
            )
            self._conn.execute(self._DDL)
            self._load_today()

    # ---- public API ----------------------------------------------------
    @property
    def daily_token_budget(self) -> int:
        return self._budget

    @property
    def unlimited(self) -> bool:
        return self._budget == 0

    def has_capacity(
        self, user_id: Optional[str], estimated_tokens: int
    ) -> bool:
        """Return True when the user may spend ``estimated_tokens`` more today."""
        if self.unlimited:
            return True
        key = (user_id or "__anon__", _utc_date())
        with self._lock:
            if self._in_cooldown(user_id):
                return False
            spent = self._spent.get(key, 0)
            return (spent + max(0, int(estimated_tokens))) <= self._budget

    def record_usage(
        self, user_id: Optional[str], tokens: int
    ) -> int:
        """Charge ``tokens`` against the user's daily budget. Returns new total."""
        tokens = max(0, int(tokens))
        key = (user_id or "__anon__", _utc_date())
        with self._lock:
            new_total = self._spent.get(key, 0) + tokens
            self._spent[key] = new_total
            if self._conn is not None:
                try:
                    self._conn.execute(
                        "INSERT INTO context_graph_llm_budget(user_id, day, tokens) "
                        "VALUES(?, ?, ?) ON CONFLICT(user_id, day) DO UPDATE SET "
                        "tokens = tokens + excluded.tokens",
                        (key[0], key[1], tokens),
                    )
                except Exception as exc:  # pragma: no cover - log-only
                    logger.debug("LLMBudget SQLite persist failed: %s", exc)
            if not self.unlimited and new_total >= self._budget:
                self._trip_cooldown(user_id)
        return new_total

    def spent(self, user_id: Optional[str]) -> int:
        key = (user_id or "__anon__", _utc_date())
        with self._lock:
            return self._spent.get(key, 0)

    def reset(self, user_id: Optional[str] = None) -> None:
        with self._lock:
            if user_id is None:
                self._spent.clear()
                self._cooldown_until.clear()
            else:
                today = _utc_date()
                self._spent.pop((user_id, today), None)
                self._cooldown_until.pop(user_id, None)

    # ---- internals -----------------------------------------------------
    def _trip_cooldown(self, user_id: Optional[str]) -> None:
        if self._cooldown_seconds == 0:
            return
        self._cooldown_until[user_id or "__anon__"] = (
            _monotonic() + self._cooldown_seconds
        )

    def _in_cooldown(self, user_id: Optional[str]) -> bool:
        until = self._cooldown_until.get(user_id or "__anon__")
        if until is None:
            return False
        if _monotonic() >= until:
            self._cooldown_until.pop(user_id or "__anon__", None)
            return False
        return True

    def _load_today(self) -> None:
        if self._conn is None:
            return
        try:
            today = _utc_date()
            cur = self._conn.execute(
                "SELECT user_id, tokens FROM context_graph_llm_budget WHERE day = ?",
                (today,),
            )
            for row in cur.fetchall():
                self._spent[(row[0], today)] = int(row[1] or 0)
        except Exception as exc:  # pragma: no cover - log-only
            logger.debug("LLMBudget SQLite hydrate failed: %s", exc)


__all__ = ["LLMBudget"]
