"""Phase D3 — backend-agnostic history store.

Provides a small, dialect-portable interface around the memory mutation
log so multiple pods can share state without colliding on a SQLite file
bind-mount. Implementations:

* :class:`SQLiteHistoryStore` — wraps the existing
  :class:`outhad_contextkit.memory.storage.SQLiteManager` (default; zero
  behaviour change for self-hosted single-tenant deployments).
* :class:`PostgresHistoryStore` — psycopg2-backed, points at the same
  Neon database the cloud control plane already uses
  (``server.cloud.db``). Reuses ``DATABASE_URL`` from the cloud env when
  present.

Selection happens through :func:`build_history_store` which inspects the
``history_db_path`` config field (already a string today). When the value
parses as a Postgres URL we route to the Postgres adapter; everything
else (paths, ``:memory:``) falls through to SQLite.
"""
from __future__ import annotations

import logging
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


_HISTORY_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS memory_history (
    id                  TEXT PRIMARY KEY,
    memory_id           TEXT,
    old_memory          TEXT,
    new_memory          TEXT,
    event               TEXT,
    created_at          TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ,
    is_deleted          INTEGER,
    actor_id            TEXT,
    role                TEXT,
    sensitive_blob      TEXT,
    encryption_metadata TEXT,
    privacy_level       TEXT,
    tenant_id           TEXT,
    sub_tenant_id       TEXT
);
"""

_HISTORY_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_memory_history_memory_id ON memory_history (memory_id)",
    "CREATE INDEX IF NOT EXISTS ix_memory_history_tenant_id ON memory_history (tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_memory_history_created_at ON memory_history (created_at)",
)


class HistoryStore(ABC):
    """Minimal contract every history backend must satisfy.

    The shape mirrors the existing ``SQLiteManager`` so callers in
    ``outhad_contextkit.memory.main`` keep working unchanged after the
    factory swap.
    """

    @abstractmethod
    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
        sensitive_blob: Optional[str] = None,
        encryption_metadata: Optional[str] = None,
        privacy_level: Optional[str] = None,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ) -> None: ...

    @abstractmethod
    def get_history(
        self,
        memory_id: str,
        *,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class PostgresHistoryStore(HistoryStore):
    """Postgres-backed history. psycopg2 (sync) keeps the surface
    identical to the SQLite adapter; the cloud control plane already
    pulls ``psycopg2-binary`` so no new dep.
    """

    def __init__(self, dsn: str, *, table: str = "memory_history") -> None:
        try:
            import psycopg2
            import psycopg2.extras  # noqa: F401  — registers DictCursor
        except ImportError as exc:  # pragma: no cover - dep missing
            raise ImportError(
                "PostgresHistoryStore requires psycopg2-binary. Install with "
                "`pip install -e .[cloud]` or add `psycopg2-binary` to your env."
            ) from exc

        self.dsn = self._normalise_dsn(dsn)
        self.table = table
        self._lock = threading.RLock()
        self._psycopg2 = psycopg2
        self._connection = None
        self._connect()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_dsn(dsn: str) -> str:
        """Translate async-style URLs (``postgresql+asyncpg://``) to the
        plain ``postgresql://`` form psycopg2 expects.

        Also strips the unsupported ``channel_binding`` query parameter
        Neon embeds in its pooler URLs (libpq honours ``sslmode`` only).
        """
        if dsn.startswith("postgresql+asyncpg://"):
            dsn = "postgresql://" + dsn[len("postgresql+asyncpg://") :]
        elif dsn.startswith("postgres+asyncpg://"):
            dsn = "postgresql://" + dsn[len("postgres+asyncpg://") :]

        # libpq does not understand ``channel_binding``; drop it.
        if "channel_binding=" in dsn:
            from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl

            parts = urlsplit(dsn)
            query = [
                (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                if k != "channel_binding"
            ]
            dsn = urlunsplit(parts._replace(query=urlencode(query)))
        return dsn

    def _connect(self) -> None:
        self._connection = self._psycopg2.connect(self.dsn)
        # Manage transactions explicitly — keeps semantics close to
        # SQLite's BEGIN/COMMIT pattern used elsewhere.
        self._connection.autocommit = False

    def _cursor(self):
        if self._connection is None or self._connection.closed:
            self._connect()
        return self._connection.cursor()

    def _ensure_schema(self) -> None:
        with self._lock:
            with self._cursor() as cur:
                cur.execute(_HISTORY_TABLE_DDL)
                for stmt in _HISTORY_INDEXES:
                    cur.execute(stmt)
                self._connection.commit()

    # ------------------------------------------------------------------
    # HistoryStore interface
    # ------------------------------------------------------------------
    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
        sensitive_blob: Optional[str] = None,
        encryption_metadata: Optional[str] = None,
        privacy_level: Optional[str] = None,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ) -> None:
        sql = (
            f"INSERT INTO {self.table} ("
            "id, memory_id, old_memory, new_memory, event, created_at, updated_at, "
            "is_deleted, actor_id, role, sensitive_blob, encryption_metadata, "
            "privacy_level, tenant_id, sub_tenant_id"
            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        params = (
            str(uuid.uuid4()),
            memory_id,
            old_memory,
            new_memory,
            event,
            created_at,
            updated_at,
            is_deleted,
            actor_id,
            role,
            sensitive_blob,
            encryption_metadata,
            privacy_level,
            tenant_id,
            sub_tenant_id,
        )
        with self._lock:
            try:
                with self._cursor() as cur:
                    cur.execute(sql, params)
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                logger.exception("PostgresHistoryStore.add_history failed")
                raise

    def get_history(
        self,
        memory_id: str,
        *,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses = ["memory_id = %s"]
        params: List[Any] = [memory_id]
        if tenant_id is not None:
            clauses.append("(tenant_id = %s OR tenant_id IS NULL)")
            params.append(tenant_id)
        if sub_tenant_id is not None:
            clauses.append("(sub_tenant_id = %s OR sub_tenant_id IS NULL)")
            params.append(sub_tenant_id)
        where = " AND ".join(clauses)
        sql = (
            "SELECT id, memory_id, old_memory, new_memory, event, "
            "created_at, updated_at, is_deleted, actor_id, role, "
            "sensitive_blob, encryption_metadata, privacy_level, "
            "tenant_id, sub_tenant_id "
            f"FROM {self.table} WHERE {where} "
            "ORDER BY created_at ASC NULLS FIRST, updated_at ASC NULLS FIRST"
        )
        with self._lock:
            with self._cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r[0],
                    "memory_id": r[1],
                    "old_memory": r[2],
                    "new_memory": r[3],
                    "event": r[4],
                    "created_at": r[5].isoformat() if r[5] else None,
                    "updated_at": r[6].isoformat() if r[6] else None,
                    "is_deleted": bool(r[7]) if r[7] is not None else False,
                    "actor_id": r[8],
                    "role": r[9],
                    "sensitive_blob": r[10],
                    "encryption_metadata": r[11],
                    "privacy_level": r[12],
                    "tenant_id": r[13],
                    "sub_tenant_id": r[14],
                }
            )
        return out

    def reset(self) -> None:
        with self._lock:
            try:
                with self._cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {self.table}")
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None and not self._connection.closed:
                try:
                    self._connection.close()
                except Exception:  # pragma: no cover
                    pass
            self._connection = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _looks_like_postgres_url(value: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parts = urlparse(value)
    except Exception:
        return False
    return parts.scheme.startswith(("postgres", "postgresql"))


def build_history_store(history_db_path: str):
    """Return the right history backend for ``history_db_path``.

    Detection rules:
    * ``postgres://...`` / ``postgresql://...`` / ``postgresql+asyncpg://...``
      → :class:`PostgresHistoryStore` (multi-pod safe).
    * Anything else (filesystem path, ``:memory:``, empty) →
      :class:`outhad_contextkit.memory.storage.SQLiteManager` (default,
      byte-identical to the legacy behaviour).
    """
    if _looks_like_postgres_url(history_db_path):
        logger.info("HistoryStore backend: postgres (Neon-style)")
        return PostgresHistoryStore(history_db_path)

    logger.info("HistoryStore backend: sqlite (path=%s)", history_db_path)
    # Local import to avoid a circular dependency on first module load.
    from outhad_contextkit.memory.storage import SQLiteManager

    return SQLiteManager(history_db_path)


__all__ = [
    "HistoryStore",
    "PostgresHistoryStore",
    "build_history_store",
]
