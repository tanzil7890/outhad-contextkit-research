"""Phase T1 — dataclasses for the tenant subsystem.

Kept dependency-free so importing this module is cheap. Persistence
sits in :mod:`outhad_contextkit.memory.tenant.registry`; routing sits
in :mod:`outhad_contextkit.memory.tenant.resolver`. Both consume these
dataclasses.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional


_DEFAULT_TENANT_SENTINEL = "__default__"
_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _slug(value: str, *, max_length: int = 32) -> str:
    """Stable slug for collection names / sub-tenant ids.

    Lowercase, alphanumeric + underscore. Any non-matching run is
    collapsed to a single underscore. Truncated to ``max_length``.
    Empty inputs yield ``"_"`` so the slug never produces an empty
    fragment that would corrupt downstream identifiers.
    """
    if not value:
        return "_"
    cleaned = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    if not cleaned:
        cleaned = "_"
    return cleaned[: max(1, int(max_length))]


def _slug_with_hash_suffix(value: str, *, max_length: int = 32) -> str:
    """Slug plus a 6-char SHA-256 prefix for collision resistance."""
    base = _slug(value, max_length=max(1, max_length - 7))
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:6]
    return f"{base}_{digest}"


@dataclass
class Tenant:
    """A top-level isolation boundary."""

    id: str
    name: str
    status: Literal["active", "suspended", "deleted"] = "active"
    created_at: datetime = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubTenant:
    """A child of a tenant — department, user, project, or custom kind.

    The id is derived from ``(tenant_id, kind, name)`` so re-creating a
    sub-tenant with the same triple yields the same id. Use
    :meth:`derive_id` to build it without instantiating.
    """

    id: str
    tenant_id: str
    name: str
    kind: Literal["department", "user", "project", "custom"] = "custom"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)

    @staticmethod
    def derive_id(tenant_id: str, name: str, kind: str = "custom") -> str:
        return f"{_slug(tenant_id)}::{_slug(kind)}::{_slug(name)}"


@dataclass
class RoleBinding:
    """Persisted (principal → role @ tenant scope) tuple."""

    id: int  # SQLite rowid; 0 before insert
    tenant_id: str
    sub_tenant_id: Optional[str]
    principal: str
    principal_kind: Literal["user", "agent", "run", "service"] = "user"
    role: str = "member"
    granted_at: datetime = field(default_factory=_now)
    granted_by: Optional[str] = None


@dataclass
class TenantContext:
    """Bundle the resolver consumes.

    Superset of :class:`outhad_contextkit.memory.personalized.types.RoleContext`.
    The conversion helper :meth:`to_role_context` keeps the MSPR layer
    interoperable.
    """

    tenant_id: Optional[str] = None
    sub_tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    role: Optional[str] = None

    def to_role_context(self):
        """Project to MSPR ``RoleContext`` (lazy import to avoid cycles)."""
        from outhad_contextkit.memory.personalized.types import RoleContext

        return RoleContext(
            user_id=self.user_id,
            agent_id=self.agent_id,
            run_id=self.run_id,
            tenant_id=self.tenant_id,
            role=self.role,
        )


__all__ = [
    "Tenant",
    "SubTenant",
    "RoleBinding",
    "TenantContext",
    "_DEFAULT_TENANT_SENTINEL",
    "_slug",
    "_slug_with_hash_suffix",
]
