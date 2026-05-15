"""Tenant + sub-tenant architecture for Outhad_ContextKit.

This package stays inert until ``MemoryConfig.tenant.enabled`` is True.
Only lightweight config + dataclass symbols are re-exported at module
import so ``import outhad_contextkit`` stays cheap when the feature
flag is off. Heavyweight modules (``registry``, ``resolver``) are
imported lazily from ``Memory._init_tenant`` so they never execute when
tenants are disabled.
"""
from __future__ import annotations

from outhad_contextkit.memory.tenant.config import (
    IsolationConfig,
    RegistryConfig,
    TenantConfig,
)
from outhad_contextkit.memory.tenant.types import (
    RoleBinding,
    SubTenant,
    Tenant,
    TenantContext,
)

__all__ = [
    "TenantConfig",
    "IsolationConfig",
    "RegistryConfig",
    "Tenant",
    "SubTenant",
    "RoleBinding",
    "TenantContext",
]
