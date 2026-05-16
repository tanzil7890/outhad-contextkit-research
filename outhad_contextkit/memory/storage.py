import logging
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SQLiteManager:
    # Authoritative column list the running code expects on the history
    # table. The order is the on-disk order used by ``CREATE TABLE`` in
    # ``_create_history_table`` so DDL stays consistent for fresh DBs.
    # Any column listed here that is missing on an existing DB is added
    # via ``ALTER TABLE ADD COLUMN`` at boot — the schema is therefore
    # self-healing across upgrades; no manual ``rm history.db`` ever.
    _HISTORY_COLUMNS: List[tuple] = [
        ("id", "TEXT PRIMARY KEY"),
        ("memory_id", "TEXT"),
        ("old_memory", "TEXT"),
        ("new_memory", "TEXT"),
        ("event", "TEXT"),
        ("created_at", "DATETIME"),
        ("updated_at", "DATETIME"),
        ("is_deleted", "INTEGER"),
        ("actor_id", "TEXT"),
        ("role", "TEXT"),
        # — tenant routing.
        ("tenant_id", "TEXT"),
        ("sub_tenant_id", "TEXT"),
        # PPMF — encrypted blobs + metadata.
        ("sensitive_blob", "TEXT"),
        ("encryption_metadata", "TEXT"),
        ("privacy_level", "TEXT"),
    ]

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._migrate_history_table()
        self._create_history_table()
        # — additive tenant columns. Run after the legacy
        # migration + create so we never miss the new columns on a
        # freshly bootstrapped DB.
        self._migrate_tenant_columns()
        # Final, idempotent sweep — guarantees every column the running
        # code requires exists on disk. Cheap (one PRAGMA + at most a
        # handful of ALTERs). Lets the engine self-heal across upgrades
        # without operators having to delete history.db manually.
        self._ensure_history_columns()

    def _ensure_history_columns(self) -> None:
        """Idempotent ``ALTER TABLE ADD COLUMN`` sweep."""
        with self._lock:
            cur = self.connection.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
            )
            if cur.fetchone() is None:
                return
            cur.execute("PRAGMA table_info(history)")
            present = {row[1] for row in cur.fetchall()}
            added = []
            for name, ddl in self._HISTORY_COLUMNS:
                if name in present:
                    continue
                # PRIMARY KEY can only be set at CREATE time; skip when
                # the column is already present (which it always will
                # be for the id column on a real table).
                col_ddl = (
                    f"{name} {ddl.replace(' PRIMARY KEY', '')}"
                    if "PRIMARY KEY" in ddl
                    else f"{name} {ddl}"
                )
                try:
                    cur.execute(f"ALTER TABLE history ADD COLUMN {col_ddl}")
                    added.append(name)
                except sqlite3.OperationalError as exc:
                    logger.warning(
                        "ALTER TABLE history ADD COLUMN %s failed: %s", name, exc
                    )
            self.connection.commit()
            if added:
                logger.info(
                    "history table self-heal added missing columns: %s",
                    ", ".join(added),
                )

    def _migrate_history_table(self) -> None:
        """
        If a pre-existing history table had the old group-chat columns,
        rename it, create the new schema, copy the intersecting data, then
        drop the old table.
        """
        with self._lock:
            try:
                # Start a transaction
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()

                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
                if cur.fetchone() is None:
                    self.connection.execute("COMMIT")
                    return  # nothing to migrate

                cur.execute("PRAGMA table_info(history)")
                old_cols = {row[1] for row in cur.fetchall()}

                expected_cols = {
                    "id",
                    "memory_id",
                    "old_memory",
                    "new_memory",
                    "event",
                    "created_at",
                    "updated_at",
                    "is_deleted",
                    "actor_id",
                    "role",
                }
                
                # PPMF: Check for encrypted storage columns
                ppmf_cols = {"sensitive_blob", "encryption_metadata", "privacy_level"}

                if old_cols == expected_cols:
                    # Add PPMF columns to existing table
                    logger.info("Adding PPMF encrypted storage columns to history table.")
                    for col in ppmf_cols:
                        if col not in old_cols:
                            cur.execute(f"ALTER TABLE history ADD COLUMN {col} TEXT")
                    self.connection.execute("COMMIT")
                    return
                
                if old_cols == (expected_cols | ppmf_cols):
                    self.connection.execute("COMMIT")
                    return

                logger.info("Migrating history table to new schema (no convo columns).")

                # Clean up any existing history_old table from previous failed migration
                cur.execute("DROP TABLE IF EXISTS history_old")

                # Rename the current history table
                cur.execute("ALTER TABLE history RENAME TO history_old")

                # Create the new history table with updated schema
                cur.execute(
                    """
                    CREATE TABLE history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                """
                )

                # Copy data from old table to new table
                intersecting = list(expected_cols & old_cols)
                if intersecting:
                    cols_csv = ", ".join(intersecting)
                    cur.execute(f"INSERT INTO history ({cols_csv}) SELECT {cols_csv} FROM history_old")

                # Drop the old table
                cur.execute("DROP TABLE history_old")

                # Commit the transaction
                self.connection.execute("COMMIT")
                logger.info("History table migration completed successfully.")

            except Exception as e:
                # Rollback the transaction on any error
                self.connection.execute("ROLLBACK")
                logger.error(f"History table migration failed: {e}")
                raise

    def _migrate_tenant_columns(self) -> None:
        """— add ``tenant_id`` / ``sub_tenant_id`` columns.

        Idempotent. Inspects ``PRAGMA table_info(history)`` and runs
        ``ALTER TABLE`` only when the columns are missing. Pre-existing
        rows are left with NULL values; the resolver treats NULL as
        the default tenant so legacy reads keep working.
        """
        with self._lock:
            try:
                cur = self.connection.cursor()
                cur.execute("PRAGMA table_info(history)")
                cols = {row[1] for row in cur.fetchall()}
                migrations = []
                if "tenant_id" not in cols:
                    migrations.append(
                        "ALTER TABLE history ADD COLUMN tenant_id TEXT"
                    )
                if "sub_tenant_id" not in cols:
                    migrations.append(
                        "ALTER TABLE history ADD COLUMN sub_tenant_id TEXT"
                    )
                if not migrations:
                    return
                self.connection.execute("BEGIN")
                for stmt in migrations:
                    cur.execute(stmt)
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_history_tenant "
                    "ON history(tenant_id)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_history_sub_tenant "
                    "ON history(sub_tenant_id)"
                )
                self.connection.execute("COMMIT")
                logger.info(
                    "history table tenant columns added (%s)",
                    ", ".join(migrations),
                )
            except Exception as exc:
                try:
                    self.connection.execute("ROLLBACK")
                except Exception:
                    pass
                logger.error("Tenant column migration failed: %s", exc)
                raise

    def _create_history_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id                     TEXT PRIMARY KEY,
                        memory_id              TEXT,
                        old_memory             TEXT,
                        new_memory             TEXT,
                        event                  TEXT,
                        created_at             DATETIME,
                        updated_at             DATETIME,
                        is_deleted             INTEGER,
                        actor_id               TEXT,
                        role                   TEXT,
                        sensitive_blob         TEXT,
                        encryption_metadata    TEXT,
                        privacy_level          TEXT
                    )
                """
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create history table: {e}")
                raise

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
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role,
                        sensitive_blob, encryption_metadata, privacy_level,
                        tenant_id, sub_tenant_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
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
                    ),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to add history record: {e}")
                raise

    def get_history(
        self,
        memory_id: str,
        *,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return history rows for ``memory_id``.

        ``tenant_id`` / ``sub_tenant_id`` are *optional* scope filters
        (). When omitted (the default), every row matching
        ``memory_id`` is returned — preserving the legacy contract for
        callers that have not opted into tenant-aware routing.
        """
        clauses = ["memory_id = ?"]
        params: List[Any] = [memory_id]
        if tenant_id is not None:
            # Match either an explicit tenant tag or legacy NULL rows so
            # an upgrade does not orphan pre-T4 data.
            clauses.append("(tenant_id = ? OR tenant_id IS NULL)")
            params.append(tenant_id)
        if sub_tenant_id is not None:
            clauses.append(
                "(sub_tenant_id = ? OR sub_tenant_id IS NULL)"
            )
            params.append(sub_tenant_id)
        where = " AND ".join(clauses)
        with self._lock:
            cur = self.connection.execute(
                f"""
                SELECT id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role,
                       sensitive_blob, encryption_metadata, privacy_level,
                       tenant_id, sub_tenant_id
                FROM history
                WHERE {where}
                ORDER BY created_at ASC, DATETIME(updated_at) ASC
            """,
                tuple(params),
            )
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "old_memory": r[2],
                "new_memory": r[3],
                "event": r[4],
                "created_at": r[5],
                "updated_at": r[6],
                "is_deleted": bool(r[7]),
                "actor_id": r[8],
                "role": r[9],
                "sensitive_blob": r[10],
                "encryption_metadata": r[11],
                "privacy_level": r[12],
                "tenant_id": r[13],
                "sub_tenant_id": r[14],
            }
            for r in rows
        ]

    def reset(self) -> None:
        """Drop and recreate the history table."""
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DROP TABLE IF EXISTS history")
                self.connection.execute("COMMIT")
                self._create_history_table()
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to reset history table: {e}")
                raise

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()
