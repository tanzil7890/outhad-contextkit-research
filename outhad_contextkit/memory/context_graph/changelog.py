"""Persistent change timeline for the Context-Graph Layer.

Every mutation performed by :class:`ContextGraph` — node created, updated,
archived, deleted; edge added, decayed, pruned — is appended to the SQLite
table ``context_graph_changes``. The table schema is deliberately additive
(``IF NOT EXISTS``) so enabling the CGL on an existing database never triggers
migrations or rewrites.

Phase E — Streaming change bus
--------------------------------
:meth:`subscribe` registers an in-proc callback and returns a token.
After each :meth:`append` the event is fanned out to every registered
subscriber's bounded queue; a per-subscriber daemon thread drains the queue
and invokes the callback so **writers are never blocked**.

Delivery semantics:
* At-least-once (SQLite is the source of truth; replay via ``from_event_id``).
* In-order per subscriber (single drain thread, FIFO queue).
* Backpressure: if a subscriber's queue is full the *oldest* item is evicted
  and a WARNING is logged.  The ``drop_count`` on the subscription tracks how
  many events have been silently dropped.
"""
from __future__ import annotations

import itertools
import json
import logging
import queue
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional

from outhad_contextkit.memory.context_graph.types import ChangeEvent

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_graph_changes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type   TEXT    NOT NULL,
    target_id    TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,
    user_id      TEXT,
    payload_json TEXT
)
"""

_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_ctx_changes_target "
    "ON context_graph_changes(target_id)"
)

_INDEX_TIMESTAMP = (
    "CREATE INDEX IF NOT EXISTS idx_ctx_changes_timestamp "
    "ON context_graph_changes(timestamp)"
)

_QUEUE_MAXSIZE = 10_000
_DRAIN_TIMEOUT = 0.1  # seconds — queue.get timeout in drain thread


# ---------------------------------------------------------------------------
#  Internal subscription record
# ---------------------------------------------------------------------------
@dataclass
class _Subscription:
    token: int
    callback: Callable[[ChangeEvent], None]
    event_types: Optional[FrozenSet[str]]  # None = all
    queue: "queue.Queue[Optional[ChangeEvent]]" = field(
        default_factory=lambda: queue.Queue(maxsize=_QUEUE_MAXSIZE)
    )
    drop_count: int = 0
    active: bool = True
    thread: Optional[threading.Thread] = None


# ---------------------------------------------------------------------------
#  ContextChangeLog
# ---------------------------------------------------------------------------
_token_gen = itertools.count(1)


class ContextChangeLog:
    """Append-only timeline of context-graph mutations with an in-proc pub/sub bus."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        # subscriber map: token → _Subscription
        self._sub_lock = threading.RLock()
        self._subscribers: Dict[int, _Subscription] = {}
        self._bootstrap()

    # ------------------------------------------------------------------
    #  SQLite helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _bootstrap(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(_SCHEMA)
                conn.execute(_INDEX)
                conn.execute(_INDEX_TIMESTAMP)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("Failed to bootstrap context change log: %s", exc)

    # ------------------------------------------------------------------
    #  Core write path
    # ------------------------------------------------------------------
    def append(self, event: ChangeEvent) -> int:
        """Persist *event* and fan it out to all live subscribers.

        Returns the SQLite autoincrement row-id (useful for replay anchoring).
        Returns -1 when the write fails.
        """
        row_id = -1
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO context_graph_changes "
                    "(event_type, target_id, timestamp, user_id, payload_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        event.event_type,
                        event.target_id,
                        event.timestamp.isoformat(),
                        event.user_id,
                        json.dumps(event.payload or {}),
                    ),
                )
                row_id = cur.lastrowid or -1
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("Failed to append context change event: %s", exc)
            return -1

        # Fan-out to live subscribers — never blocks writers
        self._dispatch(event)
        return row_id

    # ------------------------------------------------------------------
    #  Pub/sub bus
    # ------------------------------------------------------------------
    def subscribe(
        self,
        callback: Callable[[ChangeEvent], None],
        *,
        from_event_id: Optional[int] = None,
        event_types: Optional[Iterable[str]] = None,
    ) -> int:
        """Register *callback* as a subscriber and return a token.

        Args:
            callback: Called once per matching event. **Must not raise.**
            from_event_id: If given, replay all events with id > this value
                before streaming live events.  Pass ``0`` to replay the entire
                history.
            event_types: Optional whitelist of ``event_type`` strings. ``None``
                means accept all types.

        Returns:
            An integer token; pass it to :meth:`unsubscribe` to deregister.
        """
        token = next(_token_gen)
        et_filter: Optional[FrozenSet[str]] = (
            frozenset(event_types) if event_types is not None else None
        )
        sub = _Subscription(token=token, callback=callback, event_types=et_filter)

        # If replay requested: pre-fill the queue BEFORE registering for live
        # dispatch so that historical events come strictly before live ones.
        if from_event_id is not None:
            self._replay_into(sub, from_event_id)

        # Register for live dispatch
        with self._sub_lock:
            self._subscribers[token] = sub

        # Start drain thread (daemon — dies with the process)
        drain = threading.Thread(
            target=self._drain_sub,
            args=(sub,),
            name=f"cgl-bus-{token}",
            daemon=True,
        )
        drain.start()
        sub.thread = drain

        return token

    def unsubscribe(self, token: int) -> bool:
        """Deregister the subscriber with *token*.

        Returns True if the token was found and removed, False otherwise.
        """
        with self._sub_lock:
            sub = self._subscribers.pop(token, None)
        if sub is None:
            return False
        sub.active = False
        # Send sentinel to unblock the drain thread
        try:
            sub.queue.put_nowait(None)
        except queue.Full:
            pass
        return True

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _dispatch(self, event: ChangeEvent) -> None:
        """Fan *event* out to all live subscriber queues without blocking."""
        with self._sub_lock:
            subs = list(self._subscribers.values())
        for sub in subs:
            if not sub.active:
                continue
            if sub.event_types and event.event_type not in sub.event_types:
                continue
            self._enqueue(sub, event)

    def _enqueue(self, sub: _Subscription, event: ChangeEvent) -> None:
        """Put *event* into *sub*'s queue; drop oldest if full."""
        try:
            sub.queue.put_nowait(event)
        except queue.Full:
            # Evict oldest to make room
            try:
                sub.queue.get_nowait()
            except queue.Empty:
                pass
            sub.drop_count += 1
            logger.warning(
                "CGL change bus: subscriber %d queue full — dropped 1 event "
                "(total drops: %d). Consider increasing queue size or speeding up the callback.",
                sub.token,
                sub.drop_count,
            )
            try:
                sub.queue.put_nowait(event)
            except queue.Full:
                pass

    def _drain_sub(self, sub: _Subscription) -> None:
        """Daemon loop: drain *sub*'s queue and invoke the callback."""
        while sub.active:
            try:
                event = sub.queue.get(timeout=_DRAIN_TIMEOUT)
                if event is None:  # sentinel from unsubscribe
                    break
                try:
                    sub.callback(event)
                except Exception as exc:
                    logger.debug(
                        "CGL change bus: subscriber %d callback raised: %s",
                        sub.token,
                        exc,
                    )
            except queue.Empty:
                continue

    def _replay_into(self, sub: _Subscription, from_event_id: int) -> None:
        """Load all events with id > *from_event_id* from SQLite into *sub*'s queue."""
        last_id = from_event_id
        batch = 500
        while True:
            try:
                with self._lock, self._connect() as conn:
                    rows = conn.execute(
                        "SELECT id, event_type, target_id, timestamp, user_id, payload_json "
                        "FROM context_graph_changes "
                        "WHERE id > ? "
                        "ORDER BY id ASC "
                        "LIMIT ?",
                        (last_id, batch),
                    ).fetchall()
            except sqlite3.Error as exc:
                logger.warning("CGL replay failed: %s", exc)
                return
            if not rows:
                return
            for row in rows:
                event_id, event_type, target, ts, user_id, payload_json = row
                last_id = event_id
                try:
                    timestamp = datetime.fromisoformat(ts)
                except (TypeError, ValueError):
                    timestamp = datetime.utcnow()
                payload: Dict[str, Any] = {}
                if payload_json:
                    try:
                        payload = json.loads(payload_json)
                    except json.JSONDecodeError:
                        payload = {}
                ev = ChangeEvent(
                    event_type=event_type,
                    target_id=target,
                    timestamp=timestamp,
                    user_id=user_id,
                    payload=payload,
                )
                if sub.event_types and ev.event_type not in sub.event_types:
                    continue
                self._enqueue(sub, ev)

    # ------------------------------------------------------------------
    #  Read helpers (unchanged from pre-Phase-E)
    # ------------------------------------------------------------------
    def all_events(
        self,
        *,
        until: Optional[datetime] = None,
        batch: int = 1000,
    ) -> Iterable[ChangeEvent]:
        """Stream every event in ascending id order up to ``until`` (inclusive)."""
        cutoff: Optional[str] = until.isoformat() if until is not None else None
        last_id = 0
        while True:
            if cutoff is None:
                query = (
                    "SELECT id, event_type, target_id, timestamp, user_id, payload_json "
                    "FROM context_graph_changes "
                    "WHERE id > ? "
                    "ORDER BY id ASC "
                    "LIMIT ?"
                )
                params: Any = (last_id, int(batch))
            else:
                query = (
                    "SELECT id, event_type, target_id, timestamp, user_id, payload_json "
                    "FROM context_graph_changes "
                    "WHERE id > ? AND timestamp <= ? "
                    "ORDER BY id ASC "
                    "LIMIT ?"
                )
                params = (last_id, cutoff, int(batch))
            try:
                with self._lock, self._connect() as conn:
                    rows = conn.execute(query, params).fetchall()
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                logger.warning("Failed to stream context change log: %s", exc)
                return
            if not rows:
                return
            for row in rows:
                event_id, event_type, target, ts, user_id, payload_json = row
                last_id = event_id
                try:
                    timestamp = datetime.fromisoformat(ts)
                except (TypeError, ValueError):
                    timestamp = datetime.utcnow()
                payload: Dict[str, Any] = {}
                if payload_json:
                    try:
                        payload = json.loads(payload_json)
                    except json.JSONDecodeError:
                        payload = {}
                yield ChangeEvent(
                    event_type=event_type,
                    target_id=target,
                    timestamp=timestamp,
                    user_id=user_id,
                    payload=payload,
                )

    def recent(
        self,
        target_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ChangeEvent]:
        query = (
            "SELECT event_type, target_id, timestamp, user_id, payload_json "
            "FROM context_graph_changes "
        )
        params: Iterable[Any]
        if target_id is not None:
            query += "WHERE target_id = ? "
            params = (target_id,)
        else:
            params = ()
        query += "ORDER BY id DESC LIMIT ?"
        params = (*params, int(limit))
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(query, params)
                rows = cur.fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read context change log: %s", exc)
            return []
        events: List[ChangeEvent] = []
        for row in rows:
            event_type, target, ts, user_id, payload_json = row
            try:
                timestamp = datetime.fromisoformat(ts)
            except (TypeError, ValueError):
                timestamp = datetime.utcnow()
            payload: Dict[str, Any] = {}
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                except json.JSONDecodeError:
                    payload = {}
            events.append(
                ChangeEvent(
                    event_type=event_type,
                    target_id=target,
                    timestamp=timestamp,
                    user_id=user_id,
                    payload=payload,
                )
            )
        return events
