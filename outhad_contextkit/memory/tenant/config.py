"""Tenant + sub-tenant architecture configuration.

All fields default OFF. When ``TenantConfig.enabled`` is ``False`` (the
default) every tenant code path short-circuits so behaviour of
``Memory`` is byte-identical to the pre-feature main branch.

See ``doc_extra/TENANT_AND_DECAY/IMPLEMENTATION_GUIDE.md`` for the full
design and phase map.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IsolationConfig(BaseModel):
    """How strict to be about cross-tenant data leakage."""

    mode: Literal["filter", "collection", "physical"] = Field(
        default="filter",
        description=(
            "filter     — single shared collection; tenant_id added as a "
            "metadata filter at query time (matches MSPR F6 today). "
            "collection — one vector-store collection per tenant (T3). "
            "physical   — one DB file / Neo4j DB per tenant. Reserved; "
            "not implemented in T3."
        ),
    )
    deny_cross_tenant_writes: bool = Field(
        default=True,
        description="When True, raise on writes whose payload tenant_id "
        "differs from the resolved tenant.",
    )
    require_tenant_id_on_add: bool = Field(
        default=False,
        description="When True and tenant.enabled=True, every Memory.add "
        "without a tenant_id raises ValueError.",
    )


class RegistryConfig(BaseModel):
    """SQLite-backed tenant registry parameters."""

    sqlite_path: Optional[str] = Field(
        default=None,
        description="Override path for tenant_registry.db. None → "
        "{outhad_contextkit_dir}/tenant_registry.db.",
    )
    cache_size: int = Field(
        default=512,
        ge=0,
        description="LRU cache size for get_tenant() / has_role() lookups. "
        "0 disables caching.",
    )


class TenantConfig(BaseModel):
    """Master config for tenant + sub-tenant architecture."""

    enabled: bool = Field(
        default=False,
        description="Master switch. Off = zero behaviour change, zero imports.",
    )
    default_tenant_id: str = Field(
        default="__default__",
        description="Tenant id used when the caller does not supply one. "
        "Reserved sentinel; cannot be created via the admin API.",
    )
    sub_tenant_field: str = Field(
        default="sub_tenant_id",
        description="Metadata key holding the sub-tenant identifier on "
        "memory payloads.",
    )
    isolation: IsolationConfig = Field(default_factory=IsolationConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
