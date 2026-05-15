"""Phase T3 — `TenantResolver`.

Translates a :class:`TenantContext` into a concrete routing decision:

* ``vector_collection`` — which vector-store collection to read/write.
* ``history_filter``    — extra filter dict for SQLite history queries.
* ``cgl_filter``        — extra filter dict for CGL retrieval.

Three isolation modes:

* ``"filter"`` — single shared collection; tenant id flows as a
  metadata filter at query time. Matches MSPR F6 behaviour.
* ``"collection"`` — one vector-store collection per tenant. Slug
  derived from the tenant id; collisions resolved by appending a
  6-char SHA-256 suffix (deterministic).
* ``"physical"`` — reserved hook; resolver returns the same shape as
  ``"collection"`` but consumers may swap their backend per-tenant.

The resolver never raises. When the caller passes an empty / unknown
tenant context (or when the master switch is off), it returns the
default tenant pinned to ``base_collection`` so legacy code paths are
byte-identical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from outhad_contextkit.memory.tenant.config import TenantConfig
from outhad_contextkit.memory.tenant.types import (
    _DEFAULT_TENANT_SENTINEL,
    TenantContext,
    _slug,
    _slug_with_hash_suffix,
)

logger = logging.getLogger(__name__)

# Reserved characters / collision detection — keep slug stable for
# downstream consumers (Qdrant collection names, Pinecone indexes, ...).
_MAX_SLUG_LENGTH = 32


@dataclass
class ResolvedTenant:
    tenant_id: str
    sub_tenant_id: Optional[str]
    collection_name: str
    history_filter: Dict[str, str] = field(default_factory=dict)
    cgl_filter: Dict[str, str] = field(default_factory=dict)
    role: Optional[str] = None
    is_default: bool = False


class TenantResolver:
    """Resolve a tenant context to concrete routing decisions.

    Args:
        cfg: ``TenantConfig`` from ``MemoryConfig.tenant``.
        base_collection: Collection name from the operator's vector-store
                         config — used as the prefix when isolation
                         mode is ``"collection"`` and as the entire
                         collection name when mode is ``"filter"``.
        registry: Optional ``TenantRegistry``; required only for RBAC
                  enforcement, which the resolver does not perform
                  itself (left to the admin layer in T6).
    """

    def __init__(
        self,
        cfg: TenantConfig,
        *,
        base_collection: str,
        registry: Optional[Any] = None,
    ) -> None:
        self._cfg = cfg
        self._base = base_collection
        self._registry = registry

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def resolve(self, ctx: Optional[TenantContext]) -> ResolvedTenant:
        """Return the routing decision for ``ctx``.

        Tolerates ``None`` and empty tenant ids by falling back to the
        default tenant — the resolver is the single source of routing
        truth and must always succeed.
        """
        # Master switch off → resolver collapses to default tenant.
        if not self._cfg.enabled:
            return self._default(role=ctx.role if ctx else None)

        tenant_id = (ctx.tenant_id if ctx is not None else None) or self._cfg.default_tenant_id
        sub_tenant_id = ctx.sub_tenant_id if ctx is not None else None
        role = ctx.role if ctx is not None else None
        is_default = tenant_id == self._cfg.default_tenant_id

        mode = self._cfg.isolation.mode
        if mode == "filter" or is_default:
            # Single shared collection; query-time filter.
            collection = self._base
        else:
            collection = self._collection_for(tenant_id)

        history_filter: Dict[str, str] = {}
        cgl_filter: Dict[str, str] = {}
        if not is_default:
            history_filter["tenant_id"] = tenant_id
            cgl_filter["tenant_id"] = tenant_id
            if sub_tenant_id:
                history_filter[self._cfg.sub_tenant_field] = sub_tenant_id
                cgl_filter[self._cfg.sub_tenant_field] = sub_tenant_id

        return ResolvedTenant(
            tenant_id=tenant_id,
            sub_tenant_id=sub_tenant_id,
            collection_name=collection,
            history_filter=history_filter,
            cgl_filter=cgl_filter,
            role=role,
            is_default=is_default,
        )

    def collection_for(self, tenant_id: str) -> str:
        """Public alias used by admin tooling / tests."""
        if (
            not self._cfg.enabled
            or self._cfg.isolation.mode == "filter"
            or tenant_id == self._cfg.default_tenant_id
        ):
            return self._base
        return self._collection_for(tenant_id)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _collection_for(self, tenant_id: str) -> str:
        slug = _slug(tenant_id, max_length=_MAX_SLUG_LENGTH)
        # Collision-resistance: when the slug differs from the input
        # (e.g. dashes / spaces normalised to underscores), append a
        # 6-char SHA-256 suffix so two distinct tenants don't collapse
        # to the same collection. When the slug equals the input we
        # skip the suffix to keep human-readable names for clean ids.
        if slug != tenant_id.lower():
            slug = _slug_with_hash_suffix(tenant_id, max_length=_MAX_SLUG_LENGTH)
        return f"{self._base}__{slug}"

    def _default(self, *, role: Optional[str] = None) -> ResolvedTenant:
        return ResolvedTenant(
            tenant_id=_DEFAULT_TENANT_SENTINEL,
            sub_tenant_id=None,
            collection_name=self._base,
            history_filter={},
            cgl_filter={},
            role=role,
            is_default=True,
        )


__all__ = ["TenantResolver", "ResolvedTenant"]
