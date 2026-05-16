""" SQLite-backed tenant registry.

Three narrow tables drive every tenant-aware decision:

* ``tenants``       — top-level isolation boundary.
* ``sub_tenants``   — children of a tenant (department / user / project / custom).
* ``role_bindings`` — RBAC tuples.

Idempotent ``CREATE TABLE IF NOT EXISTS`` schema. WAL journal mode.
``check_same_thread=False`` so the same store can be shared across a
thread pool. Modelled after :class:`FeedbackStore` (MSPR F3) for
consistency.

The registry is the single writer. Cached lookups (``get_tenant`` and
``has_role``) use a TTL-less LRU sized by ``RegistryConfig.cache_size``.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from outhad_contextkit.memory.tenant.types import (
    _DEFAULT_TENANT_SENTINEL,
    RoleBinding,
    SubTenant,
    Tenant,
)

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    TEXT NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS sub_tenants (
    id             TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL,
    metadata_json  TEXT,
    created_at     TEXT NOT NULL,
    UNIQUE(tenant_id, name, kind)
);

CREATE TABLE IF NOT EXISTS role_bindings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id       TEXT NOT NULL,
    sub_tenant_id   TEXT,
    principal       TEXT NOT NULL,
    principal_kind  TEXT NOT NULL DEFAULT 'user',
    role            TEXT NOT NULL,
    granted_at      TEXT NOT NULL,
    granted_by      TEXT,
    UNIQUE(tenant_id, sub_tenant_id, principal, principal_kind, role)
);
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_st_tenant ON sub_tenants(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_rb_principal ON role_bindings(principal)",
    "CREATE INDEX IF NOT EXISTS idx_rb_tenant ON role_bindings(tenant_id)",
)


class _LRU:
    """Tiny dict-based LRU. Evicts oldest when full. Disabled when capacity=0."""

    def __init__(self, capacity: int) -> None:
        self._cap = max(0, int(capacity))
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: Any, default: Any = None) -> Any:
        if self._cap <= 0:
            return default
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: Any, value: Any) -> None:
        if self._cap <= 0:
            return
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._cap:
                self._data.popitem(last=False)

    def invalidate(self, predicate=None) -> None:
        if self._cap <= 0:
            return
        with self._lock:
            if predicate is None:
                self._data.clear()
                return
            for key in [k for k in self._data if predicate(k)]:
                self._data.pop(key, None)


class TenantRegistry:
    """SQLite-backed tenant + sub-tenant + role-binding store."""

    def __init__(self, db_path: str, *, cache_size: int = 512) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._tenant_cache = _LRU(cache_size)
        self._role_cache = _LRU(cache_size)
        self._bootstrap()

    # ------------------------------------------------------------------
    #  Connection / bootstrapping
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path, isolation_level=None, check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _bootstrap(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.executescript(_SCHEMA)
                for stmt in _INDEXES:
                    conn.execute(stmt)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("TenantRegistry bootstrap failed: %s", exc)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _row_to_tenant(row: sqlite3.Row) -> Tenant:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return Tenant(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            created_at=_parse_iso(row["created_at"]),
            metadata=metadata,
        )

    @staticmethod
    def _row_to_sub_tenant(row: sqlite3.Row) -> SubTenant:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return SubTenant(
            id=row["id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            kind=row["kind"],
            metadata=metadata,
            created_at=_parse_iso(row["created_at"]),
        )

    @staticmethod
    def _row_to_role(row: sqlite3.Row) -> RoleBinding:
        return RoleBinding(
            id=int(row["id"]),
            tenant_id=row["tenant_id"],
            sub_tenant_id=row["sub_tenant_id"],
            principal=row["principal"],
            principal_kind=row["principal_kind"],
            role=row["role"],
            granted_at=_parse_iso(row["granted_at"]),
            granted_by=row["granted_by"],
        )

    # ------------------------------------------------------------------
    #  Tenants
    # ------------------------------------------------------------------
    def create_tenant(
        self,
        *,
        id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        if not id or not name:
            raise ValueError("Tenant id and name must be non-empty")
        if id == _DEFAULT_TENANT_SENTINEL:
            raise ValueError(
                f"{_DEFAULT_TENANT_SENTINEL!r} is reserved for the default tenant"
            )
        ts = self._now()
        meta = json.dumps(metadata or {}, sort_keys=True)
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO tenants (id, name, status, created_at, metadata_json) "
                    "VALUES (?, ?, 'active', ?, ?)",
                    (id, name, ts, meta),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Tenant {id!r} already exists") from exc
        self._tenant_cache.invalidate(lambda k: k == id)
        return Tenant(
            id=id,
            name=name,
            status="active",
            created_at=_parse_iso(ts),
            metadata=metadata or {},
        )

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        if not tenant_id:
            return None
        cached = self._tenant_cache.get(tenant_id)
        if cached is not None:
            return cached
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM tenants WHERE id = ?", (tenant_id,)
                ).fetchone()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("get_tenant failed: %s", exc)
            return None
        if row is None:
            return None
        tenant = self._row_to_tenant(row)
        self._tenant_cache.set(tenant_id, tenant)
        return tenant

    def list_tenants(self, *, status: Optional[str] = None) -> List[Tenant]:
        try:
            with self._lock, self._connect() as conn:
                if status is None:
                    rows = conn.execute(
                        "SELECT * FROM tenants ORDER BY created_at ASC"
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM tenants WHERE status = ? ORDER BY created_at ASC",
                        (status,),
                    ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("list_tenants failed: %s", exc)
            return []
        return [self._row_to_tenant(r) for r in rows]

    def update_tenant(
        self,
        tenant_id: str,
        *,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tenant]:
        existing = self.get_tenant(tenant_id)
        if existing is None:
            return None
        new_status = status if status is not None else existing.status
        new_meta = (
            json.dumps(metadata, sort_keys=True)
            if metadata is not None
            else json.dumps(existing.metadata or {}, sort_keys=True)
        )
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "UPDATE tenants SET status = ?, metadata_json = ? WHERE id = ?",
                    (new_status, new_meta, tenant_id),
                )
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("update_tenant failed: %s", exc)
            return None
        self._tenant_cache.invalidate(lambda k: k == tenant_id)
        return self.get_tenant(tenant_id)

    def delete_tenant(self, tenant_id: str) -> bool:
        try:
            with self._lock, self._connect() as conn:
                # Cascade — sub_tenants + role_bindings reference tenants.id
                conn.execute(
                    "DELETE FROM sub_tenants WHERE tenant_id = ?", (tenant_id,)
                )
                conn.execute(
                    "DELETE FROM role_bindings WHERE tenant_id = ?", (tenant_id,)
                )
                cur = conn.execute(
                    "DELETE FROM tenants WHERE id = ?", (tenant_id,)
                )
                deleted = bool(cur.rowcount)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("delete_tenant failed: %s", exc)
            return False
        self._tenant_cache.invalidate(lambda k: k == tenant_id)
        self._role_cache.invalidate()
        return deleted

    # ------------------------------------------------------------------
    #  Sub-tenants
    # ------------------------------------------------------------------
    def create_sub_tenant(
        self,
        *,
        tenant_id: str,
        name: str,
        kind: str = "custom",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SubTenant:
        if not tenant_id or not name:
            raise ValueError("tenant_id and sub-tenant name must be non-empty")
        parent = self.get_tenant(tenant_id)
        if parent is None:
            raise ValueError(f"Unknown tenant {tenant_id!r}")
        st_id = SubTenant.derive_id(tenant_id, name, kind)
        ts = self._now()
        meta = json.dumps(metadata or {}, sort_keys=True)
        try:
            with self._lock, self._connect() as conn:
                # Idempotent — derive_id is deterministic so re-creating
                # with the same triple yields the same row.
                conn.execute(
                    "INSERT OR IGNORE INTO sub_tenants "
                    "(id, tenant_id, name, kind, metadata_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (st_id, tenant_id, name, kind, meta, ts),
                )
                row = conn.execute(
                    "SELECT * FROM sub_tenants WHERE id = ?", (st_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError(f"create_sub_tenant failed: {exc}") from exc
        return self._row_to_sub_tenant(row)

    def get_sub_tenant(self, sub_tenant_id: str) -> Optional[SubTenant]:
        if not sub_tenant_id:
            return None
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM sub_tenants WHERE id = ?", (sub_tenant_id,)
                ).fetchone()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("get_sub_tenant failed: %s", exc)
            return None
        return self._row_to_sub_tenant(row) if row else None

    def list_sub_tenants(self, tenant_id: str) -> List[SubTenant]:
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM sub_tenants WHERE tenant_id = ? "
                    "ORDER BY created_at ASC",
                    (tenant_id,),
                ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("list_sub_tenants failed: %s", exc)
            return []
        return [self._row_to_sub_tenant(r) for r in rows]

    def delete_sub_tenant(self, sub_tenant_id: str) -> bool:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "DELETE FROM role_bindings WHERE sub_tenant_id = ?",
                    (sub_tenant_id,),
                )
                cur = conn.execute(
                    "DELETE FROM sub_tenants WHERE id = ?", (sub_tenant_id,)
                )
                deleted = bool(cur.rowcount)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("delete_sub_tenant failed: %s", exc)
            return False
        self._role_cache.invalidate()
        return deleted

    # ------------------------------------------------------------------
    #  Role bindings
    # ------------------------------------------------------------------
    def assign_role(
        self,
        *,
        tenant_id: str,
        principal: str,
        role: str,
        principal_kind: str = "user",
        sub_tenant_id: Optional[str] = None,
        granted_by: Optional[str] = None,
    ) -> RoleBinding:
        if not (tenant_id and principal and role):
            raise ValueError("tenant_id, principal, and role are required")
        if self.get_tenant(tenant_id) is None:
            raise ValueError(f"Unknown tenant {tenant_id!r}")
        if sub_tenant_id and self.get_sub_tenant(sub_tenant_id) is None:
            raise ValueError(f"Unknown sub_tenant {sub_tenant_id!r}")
        ts = self._now()
        try:
            with self._lock, self._connect() as conn:
                # SQLite's UNIQUE constraint treats two NULLs as distinct,
                # so a UNIQUE(..., sub_tenant_id, ...) does not deduplicate
                # tenant-scoped bindings (sub_tenant_id IS NULL). We do an
                # explicit existence check first to keep the binding set
                # idempotent for both tenant- and sub-tenant scopes.
                existing = conn.execute(
                    "SELECT * FROM role_bindings "
                    "WHERE tenant_id = ? "
                    "  AND IFNULL(sub_tenant_id,'') = IFNULL(?,'') "
                    "  AND principal = ? AND principal_kind = ? AND role = ?",
                    (tenant_id, sub_tenant_id, principal, principal_kind, role),
                ).fetchone()
                if existing is not None:
                    row = existing
                else:
                    conn.execute(
                        "INSERT INTO role_bindings "
                        "(tenant_id, sub_tenant_id, principal, principal_kind, "
                        " role, granted_at, granted_by) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            tenant_id,
                            sub_tenant_id,
                            principal,
                            principal_kind,
                            role,
                            ts,
                            granted_by,
                        ),
                    )
                    row = conn.execute(
                        "SELECT * FROM role_bindings "
                        "WHERE tenant_id = ? "
                        "  AND IFNULL(sub_tenant_id,'') = IFNULL(?,'') "
                        "  AND principal = ? AND principal_kind = ? AND role = ?",
                        (
                            tenant_id,
                            sub_tenant_id,
                            principal,
                            principal_kind,
                            role,
                        ),
                    ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError(f"assign_role failed: {exc}") from exc
        self._role_cache.invalidate()
        return self._row_to_role(row)

    def revoke_role(
        self,
        *,
        tenant_id: str,
        principal: str,
        role: str,
        sub_tenant_id: Optional[str] = None,
        principal_kind: str = "user",
    ) -> bool:
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM role_bindings "
                    "WHERE tenant_id = ? "
                    "  AND IFNULL(sub_tenant_id,'') = IFNULL(?,'') "
                    "  AND principal = ? AND principal_kind = ? AND role = ?",
                    (tenant_id, sub_tenant_id, principal, principal_kind, role),
                )
                deleted = bool(cur.rowcount)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("revoke_role failed: %s", exc)
            return False
        self._role_cache.invalidate()
        return deleted

    def list_roles_for(self, principal: str) -> List[RoleBinding]:
        if not principal:
            return []
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM role_bindings WHERE principal = ? "
                    "ORDER BY granted_at ASC",
                    (principal,),
                ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("list_roles_for failed: %s", exc)
            return []
        return [self._row_to_role(r) for r in rows]

    def has_role(
        self,
        *,
        principal: str,
        tenant_id: str,
        role: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
        principal_kind: str = "user",
    ) -> bool:
        """Return True if ``principal`` has any matching binding.

        ``role=None`` means "any role"; useful for membership checks.
        Sub-tenant scope: if ``sub_tenant_id`` is provided, the lookup
        accepts both bindings scoped to that sub-tenant *and* tenant-
        level bindings (which implicitly cover children).
        """
        if not (principal and tenant_id):
            return False
        cache_key = (
            principal,
            principal_kind,
            tenant_id,
            sub_tenant_id,
            role,
        )
        cached = self._role_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        try:
            with self._lock, self._connect() as conn:
                if role is None and sub_tenant_id is None:
                    row = conn.execute(
                        "SELECT 1 FROM role_bindings "
                        "WHERE tenant_id = ? AND principal = ? "
                        "  AND principal_kind = ? LIMIT 1",
                        (tenant_id, principal, principal_kind),
                    ).fetchone()
                elif role is None:
                    row = conn.execute(
                        "SELECT 1 FROM role_bindings "
                        "WHERE tenant_id = ? AND principal = ? "
                        "  AND principal_kind = ? "
                        "  AND (sub_tenant_id IS NULL OR sub_tenant_id = ?) "
                        "LIMIT 1",
                        (tenant_id, principal, principal_kind, sub_tenant_id),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT 1 FROM role_bindings "
                        "WHERE tenant_id = ? AND principal = ? "
                        "  AND principal_kind = ? AND role = ? "
                        "  AND (sub_tenant_id IS NULL OR sub_tenant_id = ?) "
                        "LIMIT 1",
                        (
                            tenant_id,
                            principal,
                            principal_kind,
                            role,
                            sub_tenant_id,
                        ),
                    ).fetchone()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("has_role failed: %s", exc)
            return False
        result = row is not None
        self._role_cache.set(cache_key, result)
        return result

    # ------------------------------------------------------------------
    #  Admin helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:  # pragma: no cover - test convenience
        try:
            with self._lock, self._connect() as conn:
                conn.execute("DELETE FROM role_bindings")
                conn.execute("DELETE FROM sub_tenants")
                conn.execute("DELETE FROM tenants")
        except sqlite3.Error as exc:
            logger.warning("TenantRegistry.reset failed: %s", exc)
        self._tenant_cache.invalidate()
        self._role_cache.invalidate()


def _parse_iso(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


__all__ = ["TenantRegistry"]
