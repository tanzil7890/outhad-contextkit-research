"""Persistent feedback store — .

Append-only SQLite table of every ``record_feedback`` call. The store is
deliberately narrow: it has one write path (``append``) and a small set
of read aggregations (``aggregate``, ``personal_boost``). No ORM, no
migrations — just ``CREATE TABLE IF NOT EXISTS``.

Thread-safety:
* A module-level ``RLock`` guards writes.
* Reads use ``check_same_thread=False`` so callers can share a store
  across a thread pool (mirrors ``ContextChangeLog``).

Durability:
* ``PRAGMA journal_mode=WAL`` so writers and readers don't block each
  other.
* ``isolation_level=None`` + explicit ``BEGIN``/``COMMIT`` would be
  nicer for batch inserts; the current single-row append path does not
  need it.

Schema upgrades:
* New columns in a future phase use ``PRAGMA table_info`` check + an
  ``ALTER TABLE ADD COLUMN`` if missing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from outhad_contextkit.memory.personalized.types import FeedbackEvent

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id     TEXT    NOT NULL,
    user_id       TEXT,
    agent_id      TEXT,
    run_id        TEXT,
    query         TEXT,
    query_hash    TEXT,
    helpful       INTEGER NOT NULL,
    delta_applied REAL    NOT NULL,
    timestamp     TEXT    NOT NULL,
    metadata_json TEXT
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_fb_memory ON feedback_events(memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_fb_user   ON feedback_events(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_fb_qhash  ON feedback_events(query_hash)",
)


def hash_query(query: Optional[str]) -> Optional[str]:
    """Short, stable hash used as the lookup key in ``query_success`` later."""
    if not query:
        return None
    norm = query.strip().lower()
    if not norm:
        return None
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


class FeedbackStore:
    """SQLite-backed feedback store."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._bootstrap()

    # ------------------------------------------------------------------
    #  Bootstrapping
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path, isolation_level=None, check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _bootstrap(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(_SCHEMA)
                for stmt in _INDEXES:
                    conn.execute(stmt)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("FeedbackStore bootstrap failed: %s", exc)

    # ------------------------------------------------------------------
    #  Writes
    # ------------------------------------------------------------------
    def append(self, event: FeedbackEvent) -> int:
        """Persist ``event`` and return the SQLite rowid.

        Mutates ``event.id`` in place on success. Returns ``-1`` on
        write failure so callers can log without raising.
        """
        payload_json = json.dumps(event.metadata or {}, sort_keys=True)
        ts = event.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO feedback_events "
                    "(memory_id, user_id, agent_id, run_id, query, query_hash, "
                    "helpful, delta_applied, timestamp, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.memory_id,
                        event.user_id,
                        event.agent_id,
                        event.run_id,
                        event.query,
                        event.query_hash,
                        1 if event.helpful else 0,
                        float(event.delta_applied),
                        ts.isoformat(),
                        payload_json,
                    ),
                )
                row_id = int(cur.lastrowid or -1)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("FeedbackStore.append failed: %s", exc)
            return -1
        event.id = row_id
        return row_id

    # ------------------------------------------------------------------
    #  Reads
    # ------------------------------------------------------------------
    def _base_where(
        self,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
        window_days: Optional[int],
    ) -> Tuple[str, List[Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if memory_id is not None:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if window_days is not None and window_days > 0:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=int(window_days))
            ).isoformat()
            clauses.append("timestamp >= ?")
            params.append(cutoff)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def aggregate(
        self,
        memory_id: str,
        *,
        user_id: Optional[str] = None,
        window_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return ``{helpful, unhelpful, net}`` counts for ``memory_id``."""
        where, params = self._base_where(
            memory_id=memory_id,
            user_id=user_id,
            window_days=window_days,
        )
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT "
                    "  coalesce(sum(CASE WHEN helpful=1 THEN 1 ELSE 0 END), 0) AS helpful, "
                    "  coalesce(sum(CASE WHEN helpful=0 THEN 1 ELSE 0 END), 0) AS unhelpful, "
                    "  coalesce(sum(delta_applied), 0.0) AS net "
                    "FROM feedback_events" + where,
                    params,
                ).fetchone()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("FeedbackStore.aggregate failed: %s", exc)
            return {"helpful": 0.0, "unhelpful": 0.0, "net": 0.0}
        if row is None:
            return {"helpful": 0.0, "unhelpful": 0.0, "net": 0.0}
        return {
            "helpful": float(row["helpful"] or 0.0),
            "unhelpful": float(row["unhelpful"] or 0.0),
            "net": float(row["net"] or 0.0),
        }

    def personal_boost(
        self,
        memory_id: str,
        *,
        user_id: Optional[str] = None,
    ) -> float:
        """Return a personal boost in ``[-1, 1]``.

        Formula: ``tanh(net / scale)`` with ``scale = 3.0``. ``tanh`` keeps
        the output bounded and saturating so a handful of feedbacks
        dominate but an unbounded stream cannot blow past the cap.
        """
        stats = self.aggregate(memory_id, user_id=user_id)
        net = stats["net"]
        # tanh is monotone & keeps sign; a scale of 3 means ~0.3 after one
        # helpful verdict, ~0.76 after three, asymptotic to 1.
        import math

        scale = 3.0
        return max(-1.0, min(1.0, math.tanh(float(net) / scale)))

    def list_events(
        self,
        memory_id: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackEvent]:
        """Return recent events (most-recent-first). Used by tests + admin."""
        where, params = self._base_where(
            memory_id=memory_id,
            user_id=user_id,
            window_days=None,
        )
        params = list(params) + [int(max(1, limit))]
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM feedback_events"
                    + where
                    + " ORDER BY id DESC LIMIT ?",
                    params,
                ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("FeedbackStore.list_events failed: %s", exc)
            return []
        out: List[FeedbackEvent] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            out.append(
                FeedbackEvent(
                    id=int(row["id"]),
                    memory_id=row["memory_id"],
                    user_id=row["user_id"],
                    agent_id=row["agent_id"],
                    run_id=row["run_id"],
                    query=row["query"],
                    query_hash=row["query_hash"],
                    helpful=bool(row["helpful"]),
                    delta_applied=float(row["delta_applied"]),
                    timestamp=_parse_iso(row["timestamp"]),
                    metadata=metadata,
                )
            )
        return out

    def reset(self, *, user_id: Optional[str] = None) -> int:
        """Delete events (optionally filtered by ``user_id``). Returns row count."""
        where, params = self._base_where(
            memory_id=None, user_id=user_id, window_days=None
        )
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM feedback_events" + where, params
                )
                return int(cur.rowcount or 0)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("FeedbackStore.reset failed: %s", exc)
            return 0

    def close(self) -> None:  # pragma: no cover - convenience
        """No-op — every call opens its own short-lived connection."""
        return None


def _parse_iso(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


__all__ = ["FeedbackStore", "hash_query"]
