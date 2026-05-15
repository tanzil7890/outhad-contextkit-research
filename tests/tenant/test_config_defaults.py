"""Phase T1 — config + dataclass foundation tests."""
from __future__ import annotations

import pytest

from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.personalized.types import RoleContext
from outhad_contextkit.memory.tenant import (
    IsolationConfig,
    RegistryConfig,
    RoleBinding,
    SubTenant,
    Tenant,
    TenantConfig,
    TenantContext,
)
from outhad_contextkit.memory.tenant.types import (
    _slug,
    _slug_with_hash_suffix,
)


def test_tenant_disabled_by_default():
    cfg = MemoryConfig()
    assert cfg.tenant.enabled is False
    assert cfg.tenant.default_tenant_id == "__default__"
    assert cfg.tenant.sub_tenant_field == "sub_tenant_id"
    assert isinstance(cfg.tenant.isolation, IsolationConfig)
    assert isinstance(cfg.tenant.registry, RegistryConfig)


def test_isolation_defaults_to_filter_mode():
    cfg = MemoryConfig()
    assert cfg.tenant.isolation.mode == "filter"
    assert cfg.tenant.isolation.deny_cross_tenant_writes is True
    assert cfg.tenant.isolation.require_tenant_id_on_add is False


def test_isolation_mode_validates():
    with pytest.raises(Exception):
        IsolationConfig(mode="not-a-mode")


def test_registry_cache_size_must_be_non_negative():
    with pytest.raises(Exception):
        RegistryConfig(cache_size=-1)


def test_role_context_accepts_sub_tenant_id():
    ctx = RoleContext(tenant_id="acme", sub_tenant_id="finance", role="member")
    assert ctx.tenant_id == "acme"
    assert ctx.sub_tenant_id == "finance"
    assert ctx.role == "member"


def test_role_context_sub_tenant_defaults_to_none():
    assert RoleContext().sub_tenant_id is None


def test_tenant_dataclass_defaults():
    t = Tenant(id="acme", name="Acme Corp")
    assert t.status == "active"
    assert t.metadata == {}


def test_sub_tenant_id_is_deterministic():
    a = SubTenant.derive_id("acme", "finance", "department")
    b = SubTenant.derive_id("acme", "Finance", "department")
    # case-insensitive slug
    assert a == b
    # different kind → different id
    c = SubTenant.derive_id("acme", "finance", "user")
    assert c != a


def test_role_binding_defaults():
    rb = RoleBinding(id=0, tenant_id="acme", sub_tenant_id=None, principal="alice")
    assert rb.principal_kind == "user"
    assert rb.role == "member"


def test_tenant_context_to_role_context_round_trip():
    tc = TenantContext(
        tenant_id="acme", sub_tenant_id="finance", user_id="alice", role="admin"
    )
    rc = tc.to_role_context()
    assert rc.tenant_id == "acme"
    assert rc.user_id == "alice"
    assert rc.role == "admin"


def test_slug_lowercases_and_cleans():
    assert _slug("Acme Corp!") == "acme_corp"
    assert _slug("   ___   ") == "_"
    assert _slug("") == "_"


def test_slug_with_hash_suffix_is_deterministic():
    a = _slug_with_hash_suffix("acme")
    b = _slug_with_hash_suffix("acme")
    assert a == b
    # different inputs → different suffix
    assert _slug_with_hash_suffix("acme") != _slug_with_hash_suffix("ACME ")


def test_no_lazy_import_when_disabled():
    """tenant.* heavyweight modules must not be loaded when feature is off."""
    import sys

    # Force fresh state of registry / resolver checks.
    sys.modules.pop("outhad_contextkit.memory.tenant.registry", None)
    sys.modules.pop("outhad_contextkit.memory.tenant.resolver", None)

    cfg = MemoryConfig()
    assert cfg.tenant.enabled is False
    # Importing the package alone must NOT pull registry / resolver.
    import outhad_contextkit.memory.tenant  # noqa: F401

    assert "outhad_contextkit.memory.tenant.registry" not in sys.modules
    assert "outhad_contextkit.memory.tenant.resolver" not in sys.modules
