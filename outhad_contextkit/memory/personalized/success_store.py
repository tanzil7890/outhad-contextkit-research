""" Historical (query, memory) success store.

When a user verdict arrives via ``record_feedback``, we already log the
event to ``feedback_events.db``. F7 adds a second narrow table that maps
``(query_hash, memory_id) → helpful_count / total_count`` so future
queries with the same (or similar) hash can boost memories that
historically performed well for that query shape.

Schema (one row per verdict; aggregations are computed on read):

    query_success(id, query_hash, memory_id, user_id, helpful, timestamp)

Reads:
* ``hit_rate(query_hash, memory_id)`` — Optional[float] in [0,1]
  (None when sample size below ``min_samples``).
* ``aggregate(query_hash, memory_id)`` — full ``{helpful, total}`` row.

Thread-safe; WAL journal mode; idempotent schema bootstrap.

Industry-standard design choices:
* Append-only writes, no upserts — keeps history queryable for audits.
* SHA-256 (default) for the query hash; MinHash variant deferred but
  the store accepts whatever string the caller hands it.
* Separate ``user_id`` column so a tenant can compute personal vs
  global success scores from the same table.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_success (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash  TEXT    NOT NULL,
    memory_id   TEXT    NOT NULL,
    user_id     TEXT,
    helpful     INTEGER NOT NULL,
    timestamp   TEXT    NOT NULL
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_qs_qhash ON query_success(query_hash)",
    "CREATE INDEX IF NOT EXISTS idx_qs_mem   ON query_success(memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_qs_user  ON query_success(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_qs_pair  ON query_success(query_hash, memory_id)",
)


class QuerySuccessStore:
    """SQLite-backed (query_hash, memory_id) success table."""

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
            logger.warning("QuerySuccessStore bootstrap failed: %s", exc)

    # ------------------------------------------------------------------
    #  Writes
    # ------------------------------------------------------------------
    def record(
        self,
        *,
        query_hash: str,
        memory_id: str,
        helpful: bool,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Append a verdict row; returns SQLite rowid (-1 on failure).

        ``query_hash`` is opaque — the caller is responsible for choosing
        SHA-256, MinHash, or any other stable key. Empty / None hashes are
        rejected (no-op return -1) since they would corrupt the index.
        """
        if not query_hash or not memory_id:
            return -1
        ts = timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO query_success "
                    "(query_hash, memory_id, user_id, helpful, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        str(query_hash),
                        str(memory_id),
                        user_id,
                        1 if helpful else 0,
                        ts.isoformat(),
                    ),
                )
                return int(cur.lastrowid or -1)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("QuerySuccessStore.record failed: %s", exc)
            return -1

    # ------------------------------------------------------------------
    #  Reads
    # ------------------------------------------------------------------
    def _where(
        self,
        *,
        query_hash: Optional[str],
        memory_id: Optional[str],
        user_id: Optional[str],
        window_days: Optional[int],
    ) -> Tuple[str, List[Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if query_hash is not None:
            clauses.append("query_hash = ?")
            params.append(query_hash)
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
        *,
        query_hash: str,
        memory_id: str,
        user_id: Optional[str] = None,
        window_days: Optional[int] = None,
    ) -> Dict[str, int]:
        """Return ``{helpful, unhelpful, total}`` for the pair."""
        where, params = self._where(
            query_hash=query_hash,
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
                    "  coalesce(count(*), 0) AS total "
                    "FROM query_success" + where,
                    params,
                ).fetchone()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("QuerySuccessStore.aggregate failed: %s", exc)
            return {"helpful": 0, "unhelpful": 0, "total": 0}
        if row is None:
            return {"helpful": 0, "unhelpful": 0, "total": 0}
        return {
            "helpful": int(row["helpful"] or 0),
            "unhelpful": int(row["unhelpful"] or 0),
            "total": int(row["total"] or 0),
        }

    def hit_rate(
        self,
        query_hash: str,
        memory_id: str,
        *,
        user_id: Optional[str] = None,
        min_samples: int = 3,
        window_days: Optional[int] = None,
    ) -> Optional[float]:
        """Return ``helpful / total`` in ``[0, 1]`` or ``None``.

        ``None`` means "not enough data" — the caller treats it as a 0
        contribution to the ζ·success term so under-sampled pairs don't
        skew rankings until they have at least ``min_samples`` verdicts.
        """
        if not query_hash or not memory_id:
            return None
        stats = self.aggregate(
            query_hash=query_hash,
            memory_id=memory_id,
            user_id=user_id,
            window_days=window_days,
        )
        total = stats["total"]
        if total < int(max(1, min_samples)):
            return None
        helpful = stats["helpful"]
        return max(0.0, min(1.0, float(helpful) / float(total)))

    def top_memories(
        self,
        query_hash: str,
        *,
        user_id: Optional[str] = None,
        limit: int = 10,
        min_samples: int = 1,
    ) -> List[Tuple[str, float, int]]:
        """Return ``[(memory_id, hit_rate, total)]`` sorted by hit_rate desc.

        Used by future paraphrase/recall extensions; included now so the
        store has parity with industry analytics tables.
        """
        where, params = self._where(
            query_hash=query_hash,
            memory_id=None,
            user_id=user_id,
            window_days=None,
        )
        params = list(params) + [int(max(1, min_samples)), int(max(1, limit))]
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT memory_id, "
                    "  sum(CASE WHEN helpful=1 THEN 1 ELSE 0 END) AS helpful, "
                    "  count(*) AS total "
                    "FROM query_success"
                    + where
                    + " GROUP BY memory_id HAVING total >= ? "
                    "ORDER BY (1.0*helpful/total) DESC, total DESC LIMIT ?",
                    params,
                ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("QuerySuccessStore.top_memories failed: %s", exc)
            return []
        out: List[Tuple[str, float, int]] = []
        for r in rows:
            total = int(r["total"] or 0)
            helpful = int(r["helpful"] or 0)
            rate = float(helpful) / float(total) if total else 0.0
            out.append((str(r["memory_id"]), rate, total))
        return out

    def reset(
        self,
        *,
        user_id: Optional[str] = None,
        query_hash: Optional[str] = None,
    ) -> int:
        """Delete rows matching the supplied filter; returns row count."""
        where, params = self._where(
            query_hash=query_hash,
            memory_id=None,
            user_id=user_id,
            window_days=None,
        )
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM query_success" + where, params
                )
                return int(cur.rowcount or 0)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("QuerySuccessStore.reset failed: %s", exc)
            return 0


class SuccessProvider:
    """Per-search wrapper around ``QuerySuccessStore`` with a hot dict cache.

    The retriever calls ``hit_rate(memory_id)`` once per candidate; the
    provider caches the result so repeated lookups (e.g. across BFS
    expansion + rerank) cost zero extra SQLite reads. Cache lives for
    one search only — never shared across calls.

    Args:
        store: A ``QuerySuccessStore`` instance.
        query_hash: Stable hash for the query under evaluation. When
                    None or empty the provider always returns 0.0 so
                    the ζ term is a no-op.
        user_id: Optional scope filter.
        min_samples: Minimum verdicts before the boost activates.
    """

    def __init__(
        self,
        store: Optional[QuerySuccessStore],
        *,
        query_hash: Optional[str],
        user_id: Optional[str] = None,
        min_samples: int = 3,
    ) -> None:
        self._store = store
        self._query_hash = query_hash or None
        self._user_id = user_id
        self._min_samples = max(1, int(min_samples))
        self._cache: Dict[str, float] = {}

    @property
    def query_hash(self) -> Optional[str]:
        return self._query_hash

    def hit_rate(self, memory_id: str) -> float:
        """Return cached hit-rate for ``memory_id`` (0.0 when undefined)."""
        if not memory_id or not self._query_hash or self._store is None:
            return 0.0
        if memory_id in self._cache:
            return self._cache[memory_id]
        try:
            rate = self._store.hit_rate(
                self._query_hash,
                memory_id,
                user_id=self._user_id,
                min_samples=self._min_samples,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("SuccessProvider.hit_rate failed: %s", exc)
            rate = None
        value = float(rate) if rate is not None else 0.0
        value = max(0.0, min(1.0, value))
        self._cache[memory_id] = value
        return value


__all__ = ["QuerySuccessStore", "SuccessProvider"]
