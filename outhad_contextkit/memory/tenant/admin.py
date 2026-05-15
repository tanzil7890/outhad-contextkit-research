"""Phase T6 — public ``TenantAdmin`` facade.

Surfaced on ``Memory.tenant`` when ``MemoryConfig.tenant.enabled`` is
True. Wraps :class:`TenantRegistry` plus tenant lifecycle (T7) so ops
tooling and dashboards have one stable entry point.

The facade:
* delegates straight CRUD calls to ``TenantRegistry``
* enriches them with telemetry events (T8)
* exposes ``delete_tenant(hard=True)`` that cascades through every
  storage layer (vector store + history + CGL + registry)

Methods raise ``RuntimeError`` when MSPR/Tenant config is disabled —
silent no-ops on a misconfigured Memory would mask typos.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from outhad_contextkit.memory.telemetry import capture_event
from outhad_contextkit.memory.tenant.types import (
    RoleBinding,
    SubTenant,
    Tenant,
)

logger = logging.getLogger(__name__)


class TenantUnavailableError(RuntimeError):
    """Raised when the requested tenant is suspended / soft-deleted."""


class TenantAdmin:
    """Public admin surface for the tenant subsystem.

    Args:
        memory: A ``Memory`` instance. The admin reads
                ``memory._tenant_registry`` (T2) lazily and triggers
                lifecycle ops on ``memory`` (T7).
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _registry(self):
        registry = getattr(self._memory, "_tenant_registry", None)
        if registry is None:
            raise RuntimeError(
                "Tenant subsystem is not enabled on this Memory instance. "
                "Set MemoryConfig.tenant.enabled = True before calling "
                "Memory.tenant.*"
            )
        return registry

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        """Best-effort telemetry; never raises."""
        try:
            capture_event(event, self._memory, dict(payload))
        except Exception as exc:  # pragma: no cover - telemetry never fatal
            logger.debug("TenantAdmin telemetry emit failed: %s", exc)

    # ------------------------------------------------------------------
    # Tenants
    # ------------------------------------------------------------------
    def create_tenant(
        self,
        *,
        id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """Create a new top-level tenant. Raises ``ValueError`` if the
        id already exists or matches the reserved default sentinel."""
        tenant = self._registry().create_tenant(
            id=id, name=name, metadata=metadata
        )
        self._emit(
            "outhad_contextkit.tenant.created",
            {"tenant_id": tenant.id, "name": tenant.name},
        )
        return tenant

    def list_tenants(self, *, status: Optional[str] = None) -> List[Tenant]:
        return self._registry().list_tenants(status=status)

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        return self._registry().get_tenant(tenant_id)

    def update_tenant(
        self,
        tenant_id: str,
        *,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tenant]:
        return self._registry().update_tenant(
            tenant_id, status=status, metadata=metadata
        )

    def delete_tenant(
        self, tenant_id: str, *, hard: bool = False
    ) -> Dict[str, int]:
        """Soft- or hard-delete a tenant.

        ``hard=False`` (default) → status flipped to ``"deleted"``;
        retains every storage row so recovery is possible. The
        resolver will refuse subsequent reads.

        ``hard=True`` → cascades through every storage layer:
        vector-store collection, history rows, CGL nodes/edges, then
        registry rows (tenants + sub_tenants + role_bindings).

        Returns counts dict with the keys touched by the operation.
        """
        if hard:
            counts = self._memory._tenant_hard_delete(tenant_id)
            self._registry().delete_tenant(tenant_id)
            counts["tenant"] = 1
            self._emit(
                "outhad_contextkit.tenant.deleted",
                {"tenant_id": tenant_id, "hard": True, **counts},
            )
            return counts
        # Soft delete — flip status, archive nodes, leave storage rows.
        registry = self._registry()
        if registry.get_tenant(tenant_id) is None:
            return {"history": 0, "nodes": 0, "edges": 0, "tenant": 0}
        registry.update_tenant(tenant_id, status="deleted")
        archived = self._memory._tenant_soft_delete(tenant_id)
        self._emit(
            "outhad_contextkit.tenant.deleted",
            {"tenant_id": tenant_id, "hard": False, **archived},
        )
        return {**archived, "tenant": 1}

    # ------------------------------------------------------------------
    # Sub-tenants
    # ------------------------------------------------------------------
    def create_sub_tenant(
        self,
        *,
        tenant_id: str,
        name: str,
        kind: str = "custom",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SubTenant:
        return self._registry().create_sub_tenant(
            tenant_id=tenant_id, name=name, kind=kind, metadata=metadata
        )

    def list_sub_tenants(self, tenant_id: str) -> List[SubTenant]:
        return self._registry().list_sub_tenants(tenant_id)

    def get_sub_tenant(self, sub_tenant_id: str) -> Optional[SubTenant]:
        return self._registry().get_sub_tenant(sub_tenant_id)

    def delete_sub_tenant(self, sub_tenant_id: str) -> bool:
        return self._registry().delete_sub_tenant(sub_tenant_id)

    # ------------------------------------------------------------------
    # Role bindings
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
        binding = self._registry().assign_role(
            tenant_id=tenant_id,
            principal=principal,
            role=role,
            principal_kind=principal_kind,
            sub_tenant_id=sub_tenant_id,
            granted_by=granted_by,
        )
        return binding

    def revoke_role(
        self,
        *,
        tenant_id: str,
        principal: str,
        role: str,
        sub_tenant_id: Optional[str] = None,
        principal_kind: str = "user",
    ) -> bool:
        return self._registry().revoke_role(
            tenant_id=tenant_id,
            principal=principal,
            role=role,
            sub_tenant_id=sub_tenant_id,
            principal_kind=principal_kind,
        )

    def list_roles_for(self, principal: str) -> List[RoleBinding]:
        return self._registry().list_roles_for(principal)

    def has_role(
        self,
        *,
        principal: str,
        tenant_id: str,
        role: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
        principal_kind: str = "user",
    ) -> bool:
        return self._registry().has_role(
            principal=principal,
            tenant_id=tenant_id,
            role=role,
            sub_tenant_id=sub_tenant_id,
            principal_kind=principal_kind,
        )


__all__ = ["TenantAdmin", "TenantUnavailableError"]
